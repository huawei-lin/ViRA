"""Models and data from the Hugging Face Hub, in the layout that the training and evaluation scripts read.

    python -m vira.hf model vira-9b                 # -> $VIRA_MODELS_DIR/ViRA-9B   (also vira-27b, qwen3.5-9b,
                                                    #    qwen3.8-27b, or any repo id)
    python -m vira.hf bench [--real] [--out DIR]    # ViRA-Bench (or ViRA-Bench-Real) -> DIR/items.jsonl + images
    python -m vira.hf train [--models 9b 27b] [--out DIR]
                                                    # ViRA-Train -> DIR/train_9b.parquet, DIR/train_27b.parquet,
                                                    #    DIR/dev.jsonl, DIR/dev_single.parquet, DIR/vira_train/images

Defaults: models in $VIRA_MODELS_DIR (default $VIRA_HOME/models), data in $DATA_DIR (default $VIRA_HOME/data; VIRA_HOME
default ~/vira): ``bench`` writes to $DATA_DIR/vira_bench (``--real``: vira_bench_real), ``train`` to $DATA_DIR. These are the defaults of
``scripts/launch.sh`` and ``scripts/eval.sh``, so no further path has to be set.

ViRA-Bench row -> item of ``vira.eval.run``: qid, pair_id, question, options, gold {o,a,b}, image {o,a,b} (paths
relative to DIR), image_size, core_bbox, diff_bbox, subset, kind.
ViRA-Train rows -> the training parquet of a model: the rows marked ``in_vira9b_train`` / ``in_vira27b_train``, in the
order of ``data/vira9b_train_order.txt`` / ``data/vira27b_train_order.txt`` (each row is used once, 16 rows per step,
in this order). The ``dev`` split -> dev.jsonl (contrast sets, ``vira.eval.args``) and dev_single.parquet (single
images, one row per question and budget). ``train`` downloads only the images that these rows use; ``--models`` picks
the training subsets.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .data import ORDER, to_row, unify, write_parquet

BENCH_REPO, TRAIN_REPO = "huaweilin/ViRA-Bench", "huaweilin/ViRA-Train"
MODEL_REPOS = {"vira-9b": "huaweilin/ViRA-9B", "vira-27b": "huaweilin/ViRA-27B",
               "qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.8-27b": "Qwen/Qwen3.8-27B"}
VIRA_HOME = os.environ.get("VIRA_HOME", os.path.expanduser("~/vira"))
MODELS_DIR = os.environ.get("VIRA_MODELS_DIR", os.path.join(VIRA_HOME, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(VIRA_HOME, "data"))
DISPLAY_COLUMNS = ("zoom_o", "zoom_a", "zoom_b", "preview")     # embedded display images, not needed here
DEV_BUDGETS_MP = (0.5, 1.0, 2.0, 4.0)


# ------------------------------------------------------------------------------------------------------------- tables
def read_rows(pattern: str) -> list:
    """Rows of the parquet shards matching `pattern`, without the display images."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    cols = [c for c in pq.read_schema(files[0]).names if c not in DISPLAY_COLUMNS]
    return pa.concat_tables([pq.read_table(f, columns=cols) for f in files]).to_pylist()


def _union(boxes) -> list:
    boxes = [b for b in boxes if b]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# -------------------------------------------------------------------------------------------------------------- bench
def bench_item(r: dict) -> dict:
    """A ViRA-Bench (or ViRA-Bench-Real) row as an item of vira.eval.run. diff_bbox is the box that the zoom metrics
    use: the object's box, or on ViRA-Bench-Real the union of the object's boxes on O, A and B."""
    box = [float(v) for v in r["bbox"]]
    return {"qid": r["qid"], "pair_id": r["qid"], "set": r["pair_id"], "question": r["question"],
            "options": list(r["options"]), "gold": {v: r[f"answer_{v}"] for v in ORDER},
            "image": {v: r[f"image_{v}"] for v in ORDER}, "image_size": [int(v) for v in r["image_size"]],
            "core_bbox": box, "diff_bbox": _union([box, r.get("bbox_a"), r.get("bbox_b")]),
            "subset": r["task"], "kind": r["kind"]}


# -------------------------------------------------------------------------------------------------------------- train
def versions_of(r: dict) -> list:
    return [v for v in ORDER if r.get(f"image_{v}")]


def train_example(r: dict, image_root: str) -> dict:
    """A ViRA-Train row as the extra_info of a training row (vira.data): a contrast set with its K = 2 or 3
    versions, or a single image (K = 1; `open` when the answer is free-form)."""
    vs = versions_of(r)
    variants = [{"name": v.upper(), "path": os.path.join(image_root, r[f"image_{v}"]), "gold": str(r[f"answer_{v}"]).strip()}
                for v in vs]
    ex = {"qid": r["qid"], "question": r["question"], "options": list(r["options"] or []), "variants": variants,
          "image_size": [int(v) for v in r["image_size"]], "x_fixed": None, "open": r["answer_type"] == "open"}
    if len(vs) > 1:
        ex.update(evidence_bbox=[float(v) for v in r["diff_bbox"]], source=r["type"])
    else:
        boxes = [[float(v) for v in b] for b in (r.get("evidence_boxes") or [])]
        box = json.loads(r.get("meta") or "{}").get("evidence_bbox") or (boxes[0] if boxes else None)
        ex.update(evidence_bbox=[float(v) for v in box] if box else None, source=r["type"], evidence_boxes=boxes or None)
    return ex


def train_rows(rows: list, order: list, image_root: str) -> list:
    """The training parquet rows of one model: the rows named in `order` (qids in training order)."""
    by_qid = {r["qid"]: r for r in rows}
    return unify([to_row(train_example(by_qid[q], image_root), i) for i, q in enumerate(order)])


def dev_items(rows: list) -> list:
    """The contrast sets of the dev split as the items of vira.eval.args (image paths relative to the image root)."""
    return [{"qid": r["qid"], "question": r["question"], "options": list(r["options"]), "versions": versions_of(r),
             "gold": {v: r[f"answer_{v}"] for v in versions_of(r)}, "images": {v: r[f"image_{v}"] for v in versions_of(r)},
             "image_size": [int(v) for v in r["image_size"]], "diff_bbox": [float(v) for v in r["diff_bbox"]],
             "source": r["type"]} for r in rows if len(versions_of(r)) > 1]


def dev_single_rows(rows: list, image_root: str, budgets_mp=DEV_BUDGETS_MP) -> list:
    """The single images of the dev split as validation rows: one per (question, budget); vira.eval.args keeps the
    budgets of the run and sets the token cap of each reply."""
    out = []
    for r in rows:
        if len(versions_of(r)) != 1:
            continue
        ex = train_example(r, image_root)
        for b in budgets_mp:
            row = to_row(dict(ex, golds={"O": ex["variants"][0]["gold"]}, x_fixed=float(b), eval=True,
                              max_new_tokens_per_turn=None), len(out))
            row["data_source"] = "vira_eval/" + ex["source"]
            out.append(row)
    return unify(out)


def image_paths(rows: list) -> list:
    return sorted({r[f"image_{v}"] for r in rows for v in versions_of(r)})


# ----------------------------------------------------------------------------------------------------------- downloads
def download_files(repo: str, paths: list, local_dir: str, workers: int = 8) -> None:
    from huggingface_hub import hf_hub_download

    todo = [p for p in paths if not os.path.exists(os.path.join(local_dir, p))]
    print(f"{repo}: {len(paths)} files, {len(todo)} to download -> {local_dir}")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda p: hf_hub_download(repo, p, repo_type="dataset", local_dir=local_dir), todo))


def cmd_model(a) -> None:
    from huggingface_hub import snapshot_download

    repo = MODEL_REPOS.get(a.name.lower(), a.name)
    out = a.out or os.path.join(MODELS_DIR, repo.split("/")[-1])
    snapshot_download(repo, local_dir=out)
    print(f"{repo} -> {out}")


def cmd_bench(a) -> None:
    from huggingface_hub import snapshot_download

    out = a.out or os.path.join(DATA_DIR, "vira_bench_real" if a.real else "vira_bench")
    sub = "real/" if a.real else ""
    snapshot_download(BENCH_REPO, repo_type="dataset", local_dir=out, allow_patterns=[f"{sub}data/*", f"{sub}images/*"])
    items = [bench_item(r) for r in read_rows(os.path.join(out, sub, "data", "test-*.parquet"))]
    with open(os.path.join(out, "items.jsonl"), "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)
    print(f"{len(items)} contrast sets -> {out}/items.jsonl (BENCH_DIR={out})")


def cmd_train(a) -> None:
    from huggingface_hub import snapshot_download

    out = a.out or DATA_DIR
    root = os.path.join(out, "vira_train")
    snapshot_download(TRAIN_REPO, repo_type="dataset", local_dir=root, allow_patterns=["data/*"])
    train = read_rows(os.path.join(root, "data", "train-*.parquet"))
    dev = read_rows(os.path.join(root, "data", "dev-*.parquet"))
    used = list(dev)
    orders = {}
    for m in a.models:
        orders[m] = open(os.path.join(root, "data", f"vira{m}_train_order.txt")).read().split()
        keep = set(orders[m])
        used += [r for r in train if r["qid"] in keep]
    download_files(TRAIN_REPO, image_paths(used), root, a.workers)
    for m, order in orders.items():
        rows = train_rows(train, order, root)
        write_parquet(rows, os.path.join(out, f"train_{m}.parquet"))
        print(f"train_{m}.parquet: {len(rows)} rows ({sum(len(r['extra_info']['variants']) > 1 for r in rows)} contrast sets)")
    with open(os.path.join(out, "dev.jsonl"), "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in dev_items(dev))
    single = dev_single_rows(dev, root)
    write_parquet(single, os.path.join(out, "dev_single.parquet"))
    print(f"dev.jsonl: {len(dev_items(dev))} contrast sets; dev_single.parquet: {len(single) // len(DEV_BUDGETS_MP)} "
          f"single images (DATA_DIR={out})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("model", help="download a model")
    p.add_argument("name", help="vira-9b | vira-27b | qwen3.5-9b | qwen3.8-27b | a repo id")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_model)
    p = sub.add_parser("bench", help="download ViRA-Bench and write its items")
    p.add_argument("--real", action="store_true", help="ViRA-Bench-Real")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("train", help="download ViRA-Train and write the training and validation files")
    p.add_argument("--models", nargs="+", default=["9b", "27b"], choices=["9b", "27b"])
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
