"""ViRAAgentLoop with the real tokenizer and processor of Qwen3.5-9B and a scripted fake LLM server, CPU only.
Runs where verl 174888f and the model are installed: VIRA_MODELS_DIR=/path/to/models pytest tests -k verl
"""

import asyncio
import os

import pytest

pytest.importorskip("verl")
MODELS = os.environ.get("VIRA_MODELS_DIR", os.path.join(os.environ.get("VIRA_HOME", os.path.expanduser("~/vira")), "models"))
MODEL = os.path.join(MODELS, "Qwen3.5-9B")
if not os.path.isdir(MODEL):
    pytest.skip("no model here", allow_module_level=True)

from PIL import Image  # noqa: E402

from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from vira import common as C  # noqa: E402
from vira import protocol as P  # noqa: E402

W, H = 6000, 4000
PATCH_BOX = [3000.0, 2000.0, 3060.0, 2040.0]            # the small patch that differs between the versions
ZOOM = [491, 487, 519, 523]                             # a box around it on the 0-1000 scale of the first view
JSON_CALL = ('<think>The sign is tiny here; I will zoom in.</think>  <tool_call>\n'
             '{{"name": "image_zoom_in_tool", "arguments": {{"bbox_2d": {box}, "label": "patch"}}}}\n</tool_call>')
XML_CALL = ("<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n{box}\n</parameter>\n</function>\n</tool_call>")


class FakeServer:
    """Replays `turns`. Like vLLM with the loop's stop token, a reply ends at </tool_call>."""

    def __init__(self, tok, turns):
        self.tok, self.turns, self.calls = tok, list(turns), []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, **kw):
        self.calls.append({"prompt_ids": list(prompt_ids), "sizes": [im.size for im in (image_data or [])],
                           "max_tokens": sampling_params.get("max_tokens")})
        text = self.turns.pop(0) if self.turns else "<answer>A</answer>"
        assert sampling_params.get("stop_token_ids") == [self.tok.convert_tokens_to_ids("</tool_call>")]
        if "</tool_call>" in text:
            text = text[: text.index("</tool_call>") + len("</tool_call>")]
            ids = self.tok.encode(text, add_special_tokens=False)
        else:
            ids = self.tok.encode(text + "<|im_end|>", add_special_tokens=False)
        return TokenOutput(token_ids=ids, log_probs=[0.0] * len(ids), num_preempted=0)


def _config():
    from hydra import compose, initialize_config_dir

    from verl.trainer import main_ppo

    over = [f"actor_rollout_ref.model.path={MODEL}", "actor_rollout_ref.rollout.multi_turn.enable=True",
            "actor_rollout_ref.rollout.multi_turn.format=vira", "actor_rollout_ref.rollout.multi_turn.max_user_turns=6",
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7", "actor_rollout_ref.rollout.prompt_length=20480",
            "actor_rollout_ref.rollout.response_length=40960", "data.image_patch_size=16",
            "+data.apply_chat_template_kwargs.enable_thinking=False"]
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(main_ppo.__file__), "config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=over)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from verl.utils import hf_processor, hf_tokenizer

    d = tmp_path_factory.mktemp("photos")
    paths = {}
    for k, col in (("O", (250, 10, 10)), ("A", (10, 250, 10)), ("B", (10, 10, 250))):
        im = Image.new("RGB", (W, H), (40, 60, 80))
        im.paste(col, tuple(int(v) for v in PATCH_BOX))
        paths[k] = str(d / f"{k}.png")
        im.save(paths[k])
    ex = {"qid": "q1", "question": "What color is the small patch?", "options": ["red", "green", "blue", "white"],
          "variants": [{"name": "O", "path": paths["O"], "gold": "A"}, {"name": "A", "path": paths["A"], "gold": "B"},
                       {"name": "B", "path": paths["B"], "gold": "C"}],
          "image_size": [W, H], "evidence_bbox": PATCH_BOX, "source": "contrast_object", "x_fixed": 1.0, "index": 0}
    return {"cfg": _config(), "tok": hf_tokenizer(MODEL), "proc": hf_processor(MODEL), "ex": ex}


def run_episode(env, turns, ex=None, session_id=1, uid="u-1", **loop_kw):
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from vira.train.agent_loop import ViRAAgentLoop

    async def go():
        tools = initialize_tools_from_config(os.path.join(os.path.dirname(__file__), "..", "vira", "train", "config", "tool_config.yaml"))
        server = FakeServer(env["tok"], turns)

        class Wrap:
            def __init__(self, t):
                self.tools = t

        loop = ViRAAgentLoop(trainer_config=DictConfigWrap(env["cfg"]), server_manager=server, tokenizer=env["tok"],
                             processor=env["proc"], dataset_cls=RLHFDataset, data_config=DictConfigWrap(env["cfg"].data),
                             hf_model_type="qwen3_5", tools=Wrap(tools), name="vira_agent", **loop_kw)
        out = await loop.run({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
                             extra_info=ex or env["ex"], uid=uid, session_id=session_id, raw_prompt=[])
        return out, server
    return asyncio.run(go())


def test_zoom_then_answer(env):
    tok = env["tok"]
    vsize = P.view_size((W, H), 1_000_000)
    out, server = run_episode(env, [JSON_CALL.format(box=ZOOM), "<think>It is green.</think> <answer>B</answer>"])
    info, v = out.extra_fields["reward_extra_info"], out.extra_fields["vira"]
    assert v["version"] == "A" and info["version"] == 1 and info["n_versions"] == 3
    assert info["n_ok_calls"] == 1 and info["acc"] == 1 and info["cov_max"] == 1.0 and info["has_box"] == 1
    assert out.reward_score == 1.0                                              # the reward is the correctness
    call = v["calls"][0]
    assert call["bbox"] == P.frame_box_to_orig(ZOOM, ((0, 0, W, H), vsize)) and call["source"] == 0
    assert call["fed"] == list(C.magnified_size(call["bbox"][2] - call["bbox"][0], call["bbox"][3] - call["bbox"][1], 1_000_000))
    # images are not resized again by the processor: tokens = (w / 32) * (h / 32)
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    assert server.calls[0]["sizes"] == [vsize] and sum(t == pad for t in out.prompt_ids) == (vsize[0] // 32) * (vsize[1] // 32)
    assert server.calls[1]["sizes"][1] == tuple(call["fed"])
    assert sum(t == pad for t in out.response_ids) == (call["fed"][0] // 32) * (call["fed"][1] // 32)
    assert all(m == 0 for t, m in zip(out.response_ids, out.response_mask) if t == pad)
    assert [c["max_tokens"] for c in server.calls] == [4096, 4096]              # the cap of each reply
    txt = tok.decode(out.prompt_ids)
    assert txt.count("# Tools") == 1 and "# How to call a tool" in txt          # the tool list is in the system prompt only
    assert "Think first, call **image_zoom_in_tool** if needed" in txt and "<think>\n\n</think>" in txt   # thinking off
    resp = tok.decode(out.response_ids)
    assert "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>observation_1:<|vision_start|>" in resp
    assert "Question: What color is the small patch?\nA. red" in resp and "</tool_response>" in resp


def test_reply_stops_at_the_tool_call(env):
    """A call that is followed by an answer in the same reply is executed; <|im_end|> is added as context."""
    tok = env["tok"]
    first = JSON_CALL.format(box=ZOOM) + "\n<answer>A</answer>"
    out, server = run_episode(env, [first, "<answer>B</answer>"])
    info = out.extra_fields["reward_extra_info"]
    assert info["n_ok_calls"] == 1 and info["acc"] == 1 and len(server.calls) == 2
    im_end, tc_end = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</tool_call>")
    i = out.response_ids.index(tc_end)
    assert out.response_mask[i] == 1 and out.response_ids[i + 1] == im_end and out.response_mask[i + 1] == 0
    assert len(out.response_logprobs) == len(out.response_ids) == len(out.response_mask)


def test_xml_call_and_a_box_on_a_crop(env):
    second = JSON_CALL.format(box=[250, 250, 750, 750]).replace('"label"', '"source": "observation_1", "label"')
    out, server = run_episode(env, [XML_CALL.format(box=ZOOM), second, "<answer>B</answer>"])
    info, calls = out.extra_fields["reward_extra_info"], out.extra_fields["vira"]["calls"]
    assert info["n_ok_calls"] == 2 and calls[1]["source"] == 1
    b0, b1 = calls[0]["bbox"], calls[1]["bbox"]
    assert b0[0] < b1[0] < b1[2] < b0[2] and b0[1] < b1[1] < b1[3] < b0[3]      # inside the first crop
    assert len(server.calls[2]["sizes"]) == 3
    assert "<tool_response>observation_2:" in env["tok"].decode(out.response_ids)


def test_bad_call_then_too_many_calls(env):
    turns = ["<tool_call>{oops</tool_call>"] + [JSON_CALL.format(box=[10 * i, 10, 10 * i + 100, 110]) for i in range(6)]
    out, server = run_episode(env, turns, session_id=0)
    info = out.extra_fields["reward_extra_info"]
    assert info["n_errors"] == 1 and info["n_calls"] == 6 and info["answered"] == 0
    assert out.reward_score == 0.0 and len(server.calls) == 7                   # no answer: reward 0
    assert "<tool_response>Error: Invalid tool call format" in env["tok"].decode(out.response_ids)


def test_versions_and_budget_of_training_rows(env):
    names = [run_episode(env, ["<answer>A</answer>"], session_id=s)[0].extra_fields["vira"]["version"] for s in range(4)]
    assert names == ["O", "A", "B", "O"]                                        # rollout i runs on version i % K
    two = dict(env["ex"], variants=env["ex"]["variants"][1:], x_fixed=None)
    outs = [run_episode(env, ["<answer>B</answer>"], ex=two, session_id=s, uid="set-7", x_lo=0.5, x_hi=4.0, x_seed=42)
            for s in range(3)]
    assert [o.extra_fields["vira"]["version"] for o, _ in outs] == ["A", "B", "A"]
    assert [o.reward_score for o, _ in outs] == [1.0, 0.0, 1.0]
    x = C.budget_mp("set-7", lo=0.5, hi=4.0, seed=42)                           # one budget for the whole set
    assert all(o.extra_fields["reward_extra_info"]["x"] == x for o, _ in outs)
    assert all(s.calls[0]["sizes"] == [P.view_size((W, H), int(round(x * 1e6)))] for _, s in outs)


def test_open_question_is_judged(env, monkeypatch):
    from vira import judge

    asked = []
    monkeypatch.setattr(judge, "judge", lambda q, gold, pred, key: asked.append((q, gold, pred, key)) or (1.0, 0.0))
    ex = dict(env["ex"], options=[], variants=[dict(env["ex"]["variants"][0], gold="red")], open=True, source="detail",
              evidence_bbox=None)
    out, _ = run_episode(env, ["<think>t</think> <answer>a red patch</answer>"], ex=ex, session_id=5, judge_key_file="/k")
    info = out.extra_fields["reward_extra_info"]
    assert asked == [("What color is the small patch?", "red", "a red patch", "/k")]
    assert out.reward_score == 1.0 and info["open"] == 1 and info["answered"] == 1 and info["has_box"] == 0
    assert out.extra_fields["vira"]["answer"] == "a red patch" and info["n_versions"] == 1
    assert "\nAnswer the question using a single word or phrase.\nThink first" in env["tok"].decode(out.prompt_ids)
    out, _ = run_episode(env, ["no answer"], ex=ex)
    assert out.reward_score == 0.0 and len(asked) == 1                          # no <answer>: not judged


def test_validation_row(env):
    """A row of data.build_eval: one version, a fixed budget, its own token cap, the full reply and the answers kept."""
    from vira.data import to_row
    from vira.eval.metrics import episode_from_rollout

    ex = dict(env["ex"], variants=[env["ex"]["variants"][1]], golds={"O": "A", "A": "B", "B": "C"}, x_fixed=4.0,
              eval=True, max_new_tokens_per_turn=2048)
    ex = to_row(ex, 0)["extra_info"]
    long_reply = "<think>" + "Looking. " * 200 + "</think> The patch is red, so the answer is A."
    out, server = run_episode(env, [JSON_CALL.format(box=ZOOM), long_reply], ex=ex, session_id=0)
    assert [c["max_tokens"] for c in server.calls] == [2048, 2048]
    assert server.calls[0]["sizes"] == [P.view_size((W, H), 4_000_000)]
    v, info = out.extra_fields["vira"], out.extra_fields["reward_extra_info"]
    assert v["version"] == "A" and v["last_reply"].endswith("the answer is A.") and len(v["last_reply"]) > 600
    assert v["golds"] == {"O": "A", "A": "B", "B": "C"} and v["options"] == env["ex"]["options"]
    assert info["acc"] == 0 and info["answered"] == 0                           # no <answer> tag: strict wrong
    e = episode_from_rollout(out.extra_fields, seq_tokens=len(out.prompt_ids) + len(out.response_ids))
    assert e["letter"] == "A" and e["gold_o"] == "A" and not e["lenient"] and e["budget"] == "4mp" and e["hit"]
    out2, _ = run_episode(env, ["<answer>B</answer>"])                          # a training row keeps no full reply
    assert "last_reply" not in out2.extra_fields["vira"]
