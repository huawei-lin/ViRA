"""The agent loop of ViRA on verl (a ToolAgentLoop): one rollout of the zoom agent on one version of a contrast set.

Contrast sets without touching verl's data path: one parquet row per set holds its K versions (the original O and the
edited variants A and B, or A and B only; a single image has K = 1), rollout.n = 12, and rollout session_id i runs on
version i % K, so each version gets n = 12 / K rollouts; all share the row's uid. The pixel budget x of a set comes
from common.budget_mp(uid). The messages follow vira.protocol: the system prompt, a user turn with the first view and
the question, and after each zoom a user turn "<tool_response>...</tool_response>" with the crop; the tool list is
NOT passed to the chat template (it is in the system prompt, as in DeepEyes). The reward, the correctness of the
answer, is computed here; the advantage in the trainer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

from vira import common as C
from vira import judge as J

from .. import protocol as P
from ..images import first_view
from . import tool_parser  # noqa: F401 - registers the "vira" tool parser

VERSION_NAMES = ("O", "A", "B")


class ViRAAgentLoop(ToolAgentLoop):
    def __init__(self, *args, x_lo: float = 0.5, x_hi: float | None = None, x_seed: int = 0,
                 train_turn_cap: int = 4096, judge_key_file: str = "", **kwargs):
        """The settings of config/agent_loop.yaml (scripts/train.sh writes a copy per run): the pixel budget is
        log-uniform on [x_lo, x_hi] MP, each reply has at most train_turn_cap new tokens, and open answers are judged
        with the API key in judge_key_file ("": the key in OPENROUTER_API_KEY)."""
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        max_mp = C.MAX_PIXELS / 1e6
        self.x_lo, self.x_hi, self.x_seed = x_lo, min(x_hi or max_mp, max_mp), x_seed
        self.train_turn_cap = int(train_turn_cap)
        self.judge_key_file = str(judge_key_file or "")
        # Generation stops at </tool_call>, so a written call is executed instead of being followed by an answer in
        # the same turn. The turn's missing <|im_end|> is then added as a context token (not trained), so the token
        # stream matches the chat template.
        self._tc_end = self.tokenizer.convert_tokens_to_ids("</tool_call>")
        self._im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.tool_parser._stop_ids = [self._tc_end]

    # ------------------------------------------------------------------ rollout setup
    def _x_mp(self, ex: dict, uid: str) -> float:
        """The pixel budget of this rollout: fixed for a validation row, drawn per contrast set in training."""
        if ex.get("x_fixed"):
            return min(float(ex["x_fixed"]), C.MAX_PIXELS / 1e6)
        return C.budget_mp(uid, lo=self.x_lo, hi=self.x_hi, seed=self.x_seed)

    async def _build(self, ex: dict, uid: str, session_id: int):
        versions = list(ex["variants"])
        vi = session_id % len(versions)                   # the version of this rollout
        var = versions[vi]
        x_mp = self._x_mp(ex, uid)
        x_pixels = int(round(x_mp * 1e6))
        W, H = int(ex["image_size"][0]), int(ex["image_size"][1])
        loop = asyncio.get_running_loop()
        view = await loop.run_in_executor(None, first_view, var["path"], x_pixels)
        options = None if ex.get("open") else list(ex["options"])
        messages = [{"role": "system", "content": P.SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image", "image": view},
                                                 {"type": "text", "text": P.question_text(ex["question"], options)}]}]
        state = {"path": var["path"], "x_mp": x_mp, "x_pixels": x_pixels, "calls": [], "n_calls": 0, "errors": 0,
                 "frames": [((0, 0, W, H), tuple(view.size))],    # images shown so far: region of the photo, size
                 "question": P.question_block(ex["question"], options),
                 "version": vi, "version_name": var.get("name", VERSION_NAMES[vi] if vi < 3 else str(vi)),
                 "n_versions": len(versions), "gold": var["gold"],
                 # validation rows (data.build_eval): the row's token cap per reply; training rows: train_turn_cap
                 "turn_cap": int(ex["max_new_tokens_per_turn"]) if ex.get("eval") and ex.get("max_new_tokens_per_turn")
                 else (self.train_turn_cap if self.train_turn_cap > 0 and not ex.get("eval") else None)}
        return messages, [view], state

    # ------------------------------------------------------------------ main loop
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        ex = dict(kwargs.get("extra_info") or {})
        uid = str(kwargs.get("uid", ex.get("qid", "")))
        session_id = int(kwargs.get("session_id", 0))
        messages, images, st = await self._build(ex, uid, session_id)
        agent_data = AgentData(messages=messages, image_data=images, video_data=None, audio_data=None,
                               mm_processor_kwargs=self._get_mm_processor_kwargs(None), metrics={},
                               request_id=self._make_request_id(priority, self.rollout_config.full_determinism),
                               tools_kwargs=kwargs.get("tools_kwargs", {}) or {})
        agent_data.vira = st
        agent_data.user_turns = 0
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = []                # the tool list lives in the system prompt
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                state = AgentState.TERMINATED

        final_text = self.tokenizer.decode(agent_data.response_ids or [], skip_special_tokens=True)
        box = ex.get("evidence_bbox")
        if ex.get("open"):
            r = await self._score_open(ex, st, final_text)
        else:
            r = P.episode_score(final_text, st["gold"], st["calls"], box)
        truncated = len(agent_data.response_mask) >= self.response_length
        # the correct answers of all versions: the row's own variants, or (validation rows) the "golds" the row carries
        golds = dict(ex.get("golds") or {(v.get("name") or (VERSION_NAMES[i] if i < 3 else str(i))): v["gold"]
                                         for i, v in enumerate(ex["variants"])})
        ok_calls = [c for c in st["calls"] if c.get("bbox")]
        info = {  # the same keys, with numeric values, for every sample (verl requires it)
            "acc": r["acc"], "answered": r["answered"], "cov_max": r["cov_max"], "cov_last": r["cov_last"],
            "n_calls": float(len(st["calls"])), "n_ok_calls": float(len(ok_calls)), "n_errors": float(st["errors"]),
            "x": float(st["x_mp"]), "version": float(st["version"]), "n_versions": float(st["n_versions"]),
            "truncated": float(truncated), "open": float(bool(ex.get("open"))), "has_box": float(box is not None),
            "judge_failed": float(r.get("judge_failed", 0.0)),
            # a wrong answer on an edited variant that is the answer of the original
            "chose_o": float(bool(r["letter"]) and st["version_name"] != "O" and r["letter"] == golds.get("O")
                             and r["letter"] != st["gold"]),
        }
        extra = dict(agent_data.extra_fields)
        extra.update({
            "reward_extra_info": info,
            "vira": {"qid": ex.get("qid"), "source": ex.get("source"), "version": st["version_name"],
                     "n_versions": st["n_versions"], "x": st["x_mp"], "letter": r["letter"], "gold": st["gold"],
                     "acc": r["acc"], "calls": [dict(c) for c in st["calls"]], "final_text": final_text[-600:],
                     "answer": str(r.get("answer", ""))[:200]},
            "turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards,
        })
        if ex.get("eval"):          # the evaluator reads the full final reply, the options and the answers again
            extra["vira"].update({"last_reply": final_text, "options": list(ex["options"]), "golds": golds})
        n_resp = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-n_resp:] if n_resp else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - n_resp]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data={"images": agent_data.image_data}, mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None,
            reward_score=r["acc"], num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics, extra_fields=extra)

    async def _score_open(self, ex: dict, st: dict, final_text: str) -> dict:
        """Open questions (single images without options): the reward is the correctness of the text inside
        <answer>, judged by GPT-4o against the reference answer (vira.judge)."""
        ans = P.open_answer(final_text)
        acc = failed = 0.0
        if ans is not None:
            loop = asyncio.get_running_loop()
            acc, failed = await loop.run_in_executor(None, J.judge, ex["question"], st["gold"], ans, self.judge_key_file)
        return {"acc": float(acc), "answered": float(ans is not None), "letter": "", "answer": ans or "",
                "judge_failed": float(failed), **P.crop_coverage(st["calls"], ex.get("evidence_bbox"))}

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any],
                                       ignore_termination: bool = False) -> AgentState:
        cap = agent_data.vira.get("turn_cap")
        if cap:                                      # at most `cap` new tokens in this turn
            room = self.response_length - len(agent_data.response_mask)
            sampling_params = {**sampling_params, "max_tokens": max(1, min(int(cap), room))}
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        ids = agent_data.response_ids or []
        if ids and ids[-1] == self._tc_end:          # stopped at </tool_call>: close the turn as the template does
            agent_data.prompt_ids.append(self._im_end)
            agent_data.response_mask.append(0)
            if agent_data.response_logprobs:
                agent_data.response_logprobs.append(0.0)
        return state

    def _build_assistant_message(self, content: str, agent_data: AgentData) -> dict[str, Any]:
        # the reply exactly as generated (the <tool_call> JSON stays in the text)
        return {"role": "assistant", "content": content or ""}

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """ToolAgentLoop's tool step with the tool result as a user turn (vira.protocol) instead of a tool turn."""
        previous_messages = list(agent_data.messages)
        call = agent_data.tool_calls[0]
        with simple_timer("tool_calls", agent_data.metrics):
            resp, _, _ = await self._call_tool(call, agent_data.tools_kwargs, agent_data)
        new_images = [im for im in (resp.image or []) if im is not None]
        st = agent_data.vira
        if new_images:
            parts = P.tool_result_parts({"type": "image", "image": new_images[0]}, len(st["frames"]) - 1, st["question"])
            msg = {"role": "user", "content": parts}
        else:
            msg = {"role": "user", "content": [{"type": "text", "text": P.tool_error_turn(resp.text or "")}]}
        self._assert_mm_supported(bool(new_images))
        agent_data.messages.append(msg)
        merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
            previous_messages, agent_data.messages, agent_data.prompt_ids, agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None, tools=[])
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        if new_images:
            agent_data.image_data.append(new_images[0])
        agent_data.user_turns += 1
        return AgentState.GENERATING
