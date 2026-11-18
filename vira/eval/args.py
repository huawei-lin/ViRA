"""Arguments of the validation inside training (named after transformers' ``TrainingArguments``) and their mapping
onto verl.

verl's v1 ``fit`` loop already validates before training (``trainer.val_before_train``) and every
``trainer.test_freq`` steps, right after the new weights are synced to vLLM; ``ViRATrainer`` replaces verl's
``_validate`` with :meth:`~vira.train.trainer.ViRAMixin.evaluate`. :class:`EvalArguments` is the one
place these settings are defined; :meth:`EvalArguments.to_verl_overrides` turns them into Hydra overrides.

CLI (used by ``scripts/train.sh``): build the eval parquet and print the overrides, one per line::

    python -m vira.eval.args --eval_dataset dev.jsonl --eval_image_root ROOT \\
        --eval_output_dir OUT --eval_steps 25 --eval_budgets_mp 1 4
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass, field


@dataclass
class EvalArguments:
    """When and how the validation set is evaluated inside training."""

    eval_strategy: str = field(default="steps", metadata={
        "help": 'When to evaluate: "steps" (every eval_steps steps, and after the last step) or "no".'})
    eval_steps: int = field(default=25, metadata={"help": "Evaluate every this many training steps."})
    eval_on_start: bool = field(default=True, metadata={
        "help": "Evaluate the untrained model before the first step (logged as step 0)."})
    eval_dataset: str | None = field(default=None, metadata={
        "help": "Validation items of contrast sets: jsonl (qid, question, options, versions, gold, images, "
                "diff_bbox)."})
    eval_image_root: str | None = field(default=None, metadata={
        "help": "Directory the image paths in eval_dataset are relative to."})
    eval_extra_parquet: str | None = field(default=None, metadata={
        "help": "Optional pre-built validation parquet appended as is (the single images of the validation set: one "
                "row per (question, budget), extra_info with eval, x_fixed, golds). Rows at budgets not in "
                "eval_budgets_mp are dropped; each row's per-turn cap is set to eval_max_new_tokens. Metrics per "
                "extra_info.source ({prefix}_detail/ etc.)."})
    eval_budgets_mp: tuple[float, ...] = field(default=(1.0, 4.0), metadata={
        "help": "Fixed pixel budgets (MP) of the first view and every crop; each version is evaluated at each."})
    eval_batch_size: int = field(default=256, metadata={
        "help": "Episodes sent to the rollout servers at a time (all GPUs together)."})
    eval_max_new_tokens: int = field(default=4096, metadata={
        "help": "Token cap of each assistant turn during validation."})
    metric_key_prefix: str = field(default="dev", metadata={
        "help": 'Prefix of the logged metrics, e.g. "dev" -> dev/set_acc_strict_1mp.'})
    eval_output_dir: str | None = field(default=None, metadata={
        "help": "Where the eval parquet, eval/step_{N}.jsonl episode files and eval_history.jsonl are written."})

    def __post_init__(self) -> None:
        if self.eval_strategy not in ("steps", "no"):
            raise ValueError(f'eval_strategy must be "steps" or "no", got {self.eval_strategy!r}')
        if self.eval_strategy == "steps" and self.eval_steps <= 0:
            raise ValueError("eval_steps must be > 0 when eval_strategy is 'steps'")
        self.eval_budgets_mp = tuple(float(b) for b in self.eval_budgets_mp)

    @property
    def enabled(self) -> bool:
        return self.eval_strategy != "no" and self.eval_dataset is not None

    @property
    def eval_parquet(self) -> str:
        if not self.eval_output_dir:
            raise ValueError("eval_output_dir is required")
        return os.path.join(self.eval_output_dir, "eval_dataset.parquet")

    def to_verl_overrides(self) -> list[str]:
        """Hydra overrides that make verl call our evaluate() on this schedule with greedy decoding."""
        if not self.enabled:
            return ["trainer.val_before_train=False", "trainer.test_freq=-1"]
        return [
            f"data.val_files={self.eval_parquet}",
            f"data.val_batch_size={self.eval_batch_size}",
            "data.validation_shuffle=False",
            f"trainer.val_before_train={self.eval_on_start}",
            f"trainer.test_freq={self.eval_steps}",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0",
            "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
            "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
            f"+vira.eval.metric_key_prefix={self.metric_key_prefix}",
            f"+vira.eval.output_dir={self.eval_output_dir}",
        ]


def _parser_for(cls) -> argparse.ArgumentParser:
    """argparse parser with one --flag per dataclass field (help from the field metadata)."""
    ap = argparse.ArgumentParser(description=cls.__doc__)
    for f in dataclasses.fields(cls):
        kw = {"help": f.metadata.get("help", ""), "default": f.default}
        if f.type in (bool, "bool"):
            kw["type"] = lambda s: str(s).lower() in ("1", "true", "yes")
        elif f.type in (int, "int"):
            kw["type"] = int
        elif "tuple" in str(f.type):
            kw.update(type=float, nargs="+")
        ap.add_argument(f"--{f.name}", **kw)
    return ap


def extra_eval_rows(path: str, budgets_mp, max_new_tokens_per_turn: int | None) -> list:
    """Rows of a pre-built validation parquet (eval_extra_parquet) at the run's budgets, with the run's per-turn token
    cap."""
    import pyarrow.parquet as pq

    keep = {round(float(b), 6) for b in budgets_mp}
    rows, dropped = [], 0
    for r in pq.read_table(path).to_pylist():          # plain lists / dicts, as the other row builders give
        ex = dict(r["extra_info"])
        if not ex.get("eval") or ex.get("x_fixed") is None:
            raise ValueError(f"{path}: rows need extra_info.eval and x_fixed (qid {ex.get('qid')})")
        if round(float(ex["x_fixed"]), 6) not in keep:
            dropped += 1
            continue
        ex["max_new_tokens_per_turn"] = max_new_tokens_per_turn
        rows.append(dict(r, extra_info=ex))
    print(f"{path}: {len(rows)} validation rows added ({dropped} at other budgets dropped)", file=sys.stderr)
    return rows


def parse_eval_args(argv: list[str] | None = None) -> EvalArguments:
    return EvalArguments(**vars(_parser_for(EvalArguments).parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    args = parse_eval_args(argv)
    if args.enabled:
        from ..data import build_eval, load_jsonl, unify, write_parquet

        rows = build_eval(load_jsonl([args.eval_dataset]), args.eval_image_root or "", args.eval_budgets_mp,
                          max_new_tokens_per_turn=args.eval_max_new_tokens)
        if args.eval_extra_parquet:
            rows += extra_eval_rows(args.eval_extra_parquet, args.eval_budgets_mp, args.eval_max_new_tokens)
        rows = unify(rows)
        write_parquet(rows, args.eval_parquet)
        print(f"wrote {len(rows)} eval rows to {args.eval_parquet}", file=sys.stderr)
    print("\n".join(args.to_verl_overrides()))


if __name__ == "__main__":
    main()
