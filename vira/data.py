"""Rows of the verl parquet files: one row per contrast set or single image.

Row: data_source, agent_name, prompt (placeholder; the agent loop builds the messages), reward_model.ground_truth,
extra_info {qid, question, options, variants [{name, path, gold}], image_size, evidence_bbox, source, x_fixed, open,
index}. `variants` are the versions of the set, O (the original) and A / B (the edited variants); a single image has
one version. `open` marks a question without options, whose answer is judged. vira.hf writes the training rows from
ViRA-Train; build_eval writes the validation rows of contrast sets from an items file (vira.eval.args).

Item (one contrast set, a line of the items jsonl): qid, question, options, versions (the ones it has: "o", "a",
"b"), gold {o,a,b} (the correct option letter of each version), images {o,a,b} (paths relative to the image root),
image_size [W, H], diff_bbox (the box of the object that differs, in pixels of the photo), source.
"""

from __future__ import annotations

import json
import os

ORDER = ("o", "a", "b")


def load_jsonl(paths) -> list:
    out = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def variants_of(it: dict, image_root: str) -> list:
    """The versions of an item in O, A, B order."""
    gold, image = it["gold"], it["images"]
    versions = [v.lower() for v in it.get("versions") or ORDER]
    out = []
    for k in ORDER:
        if k in versions and gold.get(k) and image.get(k):
            out.append({"name": k.upper(), "path": os.path.join(image_root, image[k]), "gold": str(gold[k]).strip().upper()[:1]})
    if len({v["gold"] for v in out}) != len(out):
        raise ValueError(f"{it['qid']}: versions share an answer")
    return out


def set_example(it: dict, image_root: str) -> dict:
    """One contrast set as the extra_info of a row."""
    W, H = it["image_size"]
    return {"qid": it["qid"], "question": it["question"], "options": list(it["options"]),
            "variants": variants_of(it, image_root), "image_size": [int(W), int(H)],
            "evidence_bbox": [float(v) for v in it["diff_bbox"]], "source": it.get("source") or "contrast_set",
            "x_fixed": None}


def to_row(ex: dict, index: int) -> dict:
    ex = dict(ex, index=index)
    return {"data_source": f"vira/{ex['source']}", "agent_name": "vira_agent",
            "prompt": [{"role": "user", "content": ex["question"]}],
            "reward_model": {"style": "rule", "ground_truth": json.dumps({v["name"]: v["gold"] for v in ex["variants"]})},
            "extra_info": ex}


def build_eval(items: list, image_root: str, budgets_mp=(0.5, 4.0), max_new_tokens_per_turn: int | None = 4096) -> list:
    """Validation rows of contrast sets: one per (set, version, budget), in item order. Each row has a single version
    (so the agent loop answers exactly that one), a fixed budget `x_fixed`, the correct answers of all versions
    (`golds`), `eval=True` and the token cap of each reply."""
    rows = []
    for it in items:
        ex = set_example(it, image_root)
        golds = {v["name"]: v["gold"] for v in ex["variants"]}
        for b in budgets_mp:
            for v in ex["variants"]:
                rows.append(to_row(dict(ex, variants=[v], golds=golds, x_fixed=float(b), eval=True,
                                        max_new_tokens_per_turn=max_new_tokens_per_turn), len(rows)))
    for r in rows:
        r["data_source"] = "vira_eval/" + r["extra_info"]["source"]
    return rows


def unify(rows: list) -> list:
    """One extra_info schema for a parquet file: rows are re-indexed, `open` defaults to False, and a field that only
    some rows have is None in the others."""
    union = sorted({k for r in rows for k in r["extra_info"]})
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i
        r["extra_info"].setdefault("open", False)
        for k in union:
            r["extra_info"].setdefault(k, None)
    return rows


def write_parquet(rows: list, path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
