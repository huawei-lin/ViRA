"""Synchronous verl v1 trainer with the CrossRPO advantage, step metrics and validation inside training.

Hooks:
- verl's advantage computation (trainer_base._compute_advantage): verl's GRPO function is replaced by the CrossRPO
  advantage (``vira.train.crossrpo``, ``common.crossrpo_advantages``) for this call;
- verl's ``_validate`` (called before training and every ``trainer.test_freq`` steps, after the weight sync) is
  :meth:`ViRAMixin.evaluate`: the validation rows (``data.val_files``, built by ``eval/args.py``) run through the
  same agent loop on the rollout servers already loaded, and the ``dev/`` metrics (``eval.metrics``) are logged
  in the training run at that step;
- checkpoints are fully resumable (model, optimizer, extra state); only the newest ``keep_full_last`` are kept
  whole, older ones keep their HF export.
"""

from __future__ import annotations

import collections
import contextlib
import glob
import json
import logging
import os
import time
import uuid

import numpy as np
import transfer_queue as tq

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

from vira import common as C

from ..eval.metrics import Episode, compute_metrics, episode_from_rollout
from .crossrpo import make_crossrpo_fn
from .metrics import rollout_rows, step_metrics

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _as_list(v) -> list:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


class ViRAMixin:
    """The advantage, step metrics, dumps, validation and checkpoint pruning, on top of verl's sync trainer."""

    def _cfg(self, key, default):
        cfg = self.config.get("vira", None)       # the "+vira.*" overrides of scripts/train.sh
        return cfg.get(key, default) if cfg is not None else default

    # ------------------------------------------------------------------ evaluation
    def _eval_cfg(self, key, default):
        ev = self._cfg("eval", None)
        return ev.get(key, default) if ev is not None else default

    def _validate(self) -> dict[str, float]:
        """verl's validation hook (val_before_train, every test_freq steps and the last step) -> evaluate()."""
        return self.evaluate()

    def evaluate(self, eval_dataloader=None, metric_key_prefix: str | None = None) -> dict[str, float]:
        """Run the validation set once with the current weights and return its metrics (``{prefix}/...``).

        Every episode is also written to ``<output_dir>/eval/step_{N}.jsonl`` and the metrics are appended to
        ``<output_dir>/eval_history.jsonl`` (and ``self.eval_history``); verl logs the returned dict at this step."""
        prefix = metric_key_prefix or self._eval_cfg("metric_key_prefix", "dev")
        t0 = time.time()
        episodes = self._eval_episodes(eval_dataloader if eval_dataloader is not None else self.val_dataloader)
        metrics = compute_metrics(episodes, prefix)
        metrics[f"{prefix}/n_episodes"] = float(len(episodes))
        metrics[f"{prefix}/eval_seconds"] = time.time() - t0
        self._record_eval(episodes, metrics)
        logger.warning(f"vira eval step {self.global_steps}: {len(episodes)} episodes in "
                       f"{metrics[f'{prefix}/eval_seconds']:.0f} s; "
                       + ", ".join(f"{k}={v:.1f}" for k, v in sorted(metrics.items())
                                   if "set_acc_strict" in k or "acc_all" in k or "acc_macro" in k))
        return metrics

    def _eval_episodes(self, dataloader) -> list[Episode]:
        """Generate every eval row once (greedy, rollout.val_kwargs) and normalize the outputs to episodes.
        Same TransferQueue protocol as verl's own ``_validate``; rewards are not recomputed (we only read
        ``extra_fields``)."""
        episodes: list[Episode] = []
        for batch_dict in dataloader:
            n = len(batch_dict["raw_prompt"])
            batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(n)], dtype=object)
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(n)]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)
            out, _ = self.replay_buffer.sample(global_steps=self.global_steps, partition_id="val", batch_size=n)
            data = tq.kv_batch_get(keys=out.keys, partition_id=out.partition_id,
                                   select_fields=["extra_fields", "prompts", "responses"])
            lengths = [(p.numel(), r.numel()) for p, r in zip(data["prompts"].unbind(), data["responses"].unbind(),
                                                               strict=True)]
            for extra, (n_prompt, n_resp) in zip(_as_list(data["extra_fields"]), lengths, strict=True):
                episodes.append(episode_from_rollout(extra, seq_tokens=n_prompt + n_resp, prompt_tokens=n_prompt))
            tq.kv_clear(keys=out.keys, partition_id=out.partition_id)
        return episodes

    def _record_eval(self, episodes: list[Episode], metrics: dict[str, float]) -> None:
        hist = getattr(self, "eval_history", None)
        if hist is None:
            hist = self.eval_history = []
        hist.append({"step": int(self.global_steps), **metrics})
        out_dir = self._eval_cfg("output_dir", None)
        if not out_dir:
            return
        os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)
        with open(os.path.join(out_dir, "eval", f"step_{int(self.global_steps):05d}.jsonl"), "w") as f:
            for e in episodes:
                f.write(json.dumps(e, default=str) + "\n")
        with open(os.path.join(out_dir, "eval_history.jsonl"), "a") as f:
            f.write(json.dumps(hist[-1]) + "\n")

    # ------------------------------------------------------------------ training
    def _compute_advantage(self, batch, metrics: dict):
        from verl.trainer.ppo.v1 import trainer_base as tb

        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extras = _as_list(data["extra_fields"])
        try:
            metrics.update(step_metrics(extras))
            # the same rates over the last 10 steps (a rate from the rollouts of one step is noisy): train10/*
            hist = getattr(self, "_recent", None)
            if hist is None:
                hist = self._recent = collections.deque(maxlen=10)
            hist.append([{"reward_extra_info": e.get("reward_extra_info", {}),
                          "vira": {"source": (e.get("vira") or {}).get("source")}} if isinstance(e, dict) else {}
                         for e in extras])
            metrics.update(step_metrics([e for h in hist for e in h], prefix="train10"))
        except Exception as e:  # noqa: BLE001 - metrics must never stop training
            logger.warning(f"vira metrics failed: {e}")
        if self._cfg("dump_dir", None):
            self._dump(extras)
        orig = tb.compute_advantage_for_multi_trajectories
        tb.compute_advantage_for_multi_trajectories = make_crossrpo_fn(
            rollout_rows(list(batch.keys), extras), float(self._cfg("crossrpo_lambda", C.CROSSRPO_LAMBDA)), metrics)
        try:
            return super()._compute_advantage(batch, metrics)
        finally:
            tb.compute_advantage_for_multi_trajectories = orig

    def _dump(self, extras):
        """Every rollout of the step (its calls, answer and metrics) -> <dump_dir>/step_{N}.jsonl."""
        d = self._cfg("dump_dir", None)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"step_{self.global_steps:05d}.jsonl"), "w") as f:
            for e in extras:
                row = dict(e.get("vira", {})) if isinstance(e, dict) else {}
                row["info"] = e.get("reward_extra_info", {}) if isinstance(e, dict) else {}
                f.write(json.dumps(row, default=str) + "\n")

    def _save_checkpoint(self):
        """Every checkpoint is fully resumable (model, optimizer, extra state). The newest
        ``keep_full_last`` stay so; older ones keep only their HF export. Pruning down to ``keep_full_last - 1`` runs
        before the new save, so a failed save (e.g. disk full) never touches the newest complete checkpoint and the
        disk holds at most ``keep_full_last`` full checkpoints."""
        k = max(1, int(self._cfg("keep_full_last", 2)))
        prune = self._cfg("prune_old_ckpt", True)
        if prune:
            self._prune_ckpts(max(1, k - 1), exclude=self.global_steps)
        out = super()._save_checkpoint()
        if prune:
            self._prune_ckpts(k)
        return out

    def _prune_ckpts(self, keep_full: int, exclude: int | None = None):
        import shutil

        root = self.config.trainer.default_local_dir
        steps = sorted(int(os.path.basename(d).split("_")[-1]) for d in glob.glob(os.path.join(root, "global_step_*")))
        steps = [s for s in steps if s != exclude]
        freed = 0
        for st in steps[:-keep_full]:
            d = os.path.join(root, f"global_step_{st}")
            for p in glob.glob(os.path.join(d, "actor", "*_world_size_*.pt")):  # model, optimizer, extra state shards
                freed += os.path.getsize(p)
                os.remove(p)
        k = int(self._cfg("keep_hf_last", 0))  # > 0: also drop the HF export beyond the last k checkpoints
        if k > 0:
            for st in steps[:-k]:
                hf = os.path.join(root, f"global_step_{st}", "actor", "huggingface")
                if os.path.isdir(hf):
                    freed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(hf) for f in fs)
                    shutil.rmtree(hf, ignore_errors=True)
        if freed:
            logger.warning(f"vira: pruned {freed / 1e9:.1f} GB of older checkpoint files (keeping {keep_full} full)")


@contextlib.contextmanager
def _as_sync(trainer):
    mode, trainer.trainer_mode = trainer.trainer_mode, "sync"
    try:
        yield
    finally:
        trainer.trainer_mode = mode


@register_trainer("vira_sync")
class ViRATrainer(ViRAMixin, PPOTrainerSync):
    """verl's synchronous loop with the ViRA hooks.

    verl saves and restores the TransferQueue in checkpoints for every ``trainer_mode`` other than "sync". Nothing is in
    flight at a step boundary of this trainer, so, like verl's own sync trainer, it does neither: the snapshot would
    only hold the finished batch, and after a resume nothing would ever clear it."""

    def _save_checkpoint(self):
        with _as_sync(self):
            return super()._save_checkpoint()

    def _load_checkpoint(self):
        with _as_sync(self):
            return super()._load_checkpoint()


__all__ = ["ViRATrainer"]
