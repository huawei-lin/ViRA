"""V* and HR-Bench 4K/8K as single-image items for ``vira.eval.run``.

    python -m vira.eval.ext_items RAW OUT

RAW: vstar_bench/ (craigwu/vstar_bench: test_questions.jsonl + direct_attributes/, relative_position/) and hr_bench/
(DreamMr/HR-Bench: hr_bench_{4k,8k}.parquet with base64 JPEGs). Each question becomes one item with a single view "o":
the question (options stripped from the V* text), the options in their order, the gold letter, the image file (V*: as
shipped; HR-Bench: the decoded JPEG bytes, one file per distinct image), its byte size and pixel size.
Writes OUT/items_ext.jsonl and OUT/images/... (paths relative to OUT, the --data-root of the evaluation).
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import os
import re
import shutil
import sys

import pyarrow.parquet as pq
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def item(bench, idx, question, options, gold, rel, out, category):
    p = os.path.join(out, rel)
    with Image.open(p) as im:
        size = list(im.size)
    qid = f"{bench}_{idx}"
    return {"qid": qid, "pair_id": qid, "photo_group": rel, "bench": bench, "kind": "ext", "category": category,
            "question": question, "options": options, "gold": {"o": gold}, "image": {"o": rel},
            "bytes": {"o": os.path.getsize(p)}, "image_size": size}


def main(raw, out):
    items = []
    # V*: "What is ...?\n(A) x\n(B) y\n...\nAnswer with the option's letter from the given choices directly."
    vdir = os.path.join(raw, "vstar_bench")
    for line in open(os.path.join(vdir, "test_questions.jsonl")):
        r = json.loads(line)
        lines = [x.strip() for x in r["text"].split("\n") if x.strip()]
        opts = [re.sub(r"^\([A-H]\)\s*", "", x) for x in lines if re.match(r"^\([A-H]\)", x)]
        rel = f"images/vstar/{r['image']}"
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(vdir, r["image"]), os.path.join(out, rel))
        items.append(item("vstar", r["question_id"], lines[0], opts, r["label"], rel, out, r["category"]))
    # HR-Bench 4K / 8K
    for bench, fn in (("hrbench4k", "hr_bench_4k.parquet"), ("hrbench8k", "hr_bench_8k.parquet")):
        pf = pq.ParquetFile(os.path.join(raw, "hr_bench", fn))
        seen = set()
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                key = hashlib.sha1(r["image"].encode()).hexdigest()[:16]
                rel = f"images/{bench}/{key}.jpg"
                if key not in seen:
                    os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                    with open(os.path.join(out, rel), "wb") as f:
                        f.write(base64.b64decode(r["image"]))
                    seen.add(key)
                opts = [r[k] for k in "ABCD" if r.get(k) not in (None, "")]
                items.append(item(bench, r["index"], r["question"].strip(), opts, r["answer"], rel, out, r["category"]))
    with open(os.path.join(out, "items_ext.jsonl"), "w") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    c = collections.Counter(it["bench"] for it in items)
    imgs = collections.Counter(it["bench"] for it in {it["image"]["o"]: it for it in items}.values())
    print("items", dict(c), "| distinct images", dict(imgs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
