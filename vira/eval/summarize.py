"""python -m vira.eval.summarize DIR -- markdown tables of an evaluation directory (scripts/eval.sh output).

DIR holds <label>/bench_<budget>.jsonl, <label>/ext_<budget>.jsonl and <label>/gen_<budget>.jsonl (vira.eval.run
rows; budget = 0.5mp, 1mp, 2mp, 4mp, maxmp) plus items_bench.jsonl (the items of the contrast sets, for the object
boxes and the originals' answers), items_ext.jsonl and items_gen.jsonl. Tables:
  contrast sets (ViRA-Bench or ViRA-Bench-Real): set accuracy (strict / lenient reading of the answer), per-image
  accuracy, O/A/B, answered, last crop on the object, accuracy when a crop covers the object, zooms per image, tokens
  per image, per budget;
  V*, HR-Bench 4K, HR-Bench 8K: accuracy (lenient / strict reading of the answer), per budget;
  the benchmarks of items_gen.jsonl (MME-RealWorld-Lite, TreeBench, O3-Bench, MMStar, HallusionBench), if any.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

from vira.eval.metrics import budget_metrics, episode_from_row

TAGS = ["0.5mp", "1mp", "2mp", "4mp", "maxmp"]
NAME = {"0.5mp": "0.5 MP", "1mp": "1 MP", "2mp": "2 MP", "4mp": "4 MP", "maxmp": "max"}
EXT = [("vstar", "V*"), ("hrbench4k", "HR-Bench 4K"), ("hrbench8k", "HR-Bench 8K")]
GEN = [("mmerw_lite", "MME-RW-Lite"), ("treebench", "TreeBench"), ("o3bench", "O3-Bench"), ("mmstar", "MMStar"),
       ("hallusion", "HallusionBench")]


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in map(json.loads, f) if r.get("status") != "error"]


def items(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {it["qid"]: it for it in map(json.loads, f)}


def main(d):
    labels = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p))
    items_b = items(os.path.join(d, "items_bench.jsonl"))
    print("## Contrast sets (versions O / A / B)\n")
    print("| model | budget | set acc. (strict) | set acc. (lenient) | acc | O / A / B | answered | last crop on object "
          "| acc when covered | zooms | tokens / image | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        for t in TAGS:
            rs = rows(os.path.join(d, lab, f"bench_{t}.jsonl"))
            if not rs:
                continue
            m = budget_metrics([episode_from_row(r, items_b) for r in rs], "b")
            g = lambda k: m.get(f"b/{k}_{t}", float("nan"))  # noqa: E731
            print(f"| {lab} | {NAME[t]} | {g('set_acc_strict'):.1f} | "
                  f"{g('set_acc_lenient'):.1f} | {g('acc'):.1f} | {g('acc_O'):.1f} / {g('acc_A'):.1f} / {g('acc_B'):.1f} | "
                  f"{g('answered'):.1f} | {g('hit_last'):.1f} | {g('acc_when_hit'):.1f} | {g('zooms'):.2f} | "
                  f"{g('tokens'):.0f} | {g('n_questions'):.0f} |")
    single(d, labels, "ext", EXT, "V* / HR-Bench")
    single(d, labels, "gen", GEN, "More general benchmarks")


def single(d, labels, name, benches, title):
    """The single-image benchmarks of items_<name>.jsonl (<label>/<name>_<budget>.jsonl): accuracy, lenient / strict."""
    its = items(os.path.join(d, f"items_{name}.jsonl"))
    if not its:
        return
    print(f"\n## {title} (accuracy, lenient / strict)\n")
    print("| model | budget | " + " | ".join(n for _, n in benches) + " | zooms | n |")
    print("|---|---|" + "---|" * len(benches) + "---|---|")
    for lab in labels:
        for t in ("1mp", "maxmp"):
            rs = rows(os.path.join(d, lab, f"{name}_{t}.jsonl"))
            if not rs:
                continue
            by = defaultdict(list)
            for r in rs:
                by[its[r["qid"]]["bench"]].append(episode_from_row(r, its))
            cell = lambda b: (f"{100 * sum(e['lenient'] for e in by[b]) / len(by[b]):.1f} / "  # noqa: E731
                              f"{100 * sum(e['strict'] for e in by[b]) / len(by[b]):.1f}") if by[b] else "-"
            z = sum(e["zooms"] for es in by.values() for e in es) / len(rs)
            print(f"| {lab} | {NAME[t]} | " + " | ".join(cell(b) for b, _ in benches) + f" | {z:.2f} | {len(rs)} |")


if __name__ == "__main__":
    main(sys.argv[1])
