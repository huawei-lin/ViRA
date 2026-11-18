"""More general benchmarks as single-image items for ``vira.eval.run``, in the format of ``ext_items``: fine-grained /
high-resolution (MME-RealWorld-Lite, TreeBench, O3-Bench), general (MMStar) and hallucination (HallusionBench).

    python -m vira.eval.gen_items RAW OUT

RAW holds the Hugging Face downloads, one directory each (``hf download <id> --repo-type dataset --local-dir RAW/<name>``):
  MME-RealWorld-Lite  yifanzhang114/MME-RealWorld-Lite, data.zip unzipped: data/MME-RealWorld-Lite.json, data/imgs/
  TreeBench           HaochenWang/TreeBench: TreeBench.tsv (base64 images)
  O3-Bench            m-Just/O3-Bench: test/metadata.jsonl, test/images/{chart,map}/
  MMStar              Lin-Chen/MMStar: mmstar.parquet
  HallusionBench      lmms-lab/HallusionBench: data/image-*.parquet
Yes/no questions (HallusionBench) get the options Yes / No. HallusionBench's 178 questions without an image are
left out (every item needs an image); MMStar's trailing "nan" options are dropped. Images keep their shipped bytes, one file per
distinct image. Writes OUT/items_gen.jsonl and OUT/images/<bench>/... (paths relative to OUT, the --data-root).
"""

from __future__ import annotations

import ast
import base64
import collections
import csv
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys

import pyarrow.parquet as pq
from PIL import Image

from vira.eval.ext_items import item

Image.MAX_IMAGE_PIXELS = None
csv.field_size_limit(10 ** 9)
YES_NO = ["Yes", "No"]


def save_bytes(out, bench, data: bytes) -> str:
    """Write image bytes once per distinct content; returns the path relative to OUT."""
    fmt = Image.open(io.BytesIO(data)).format.lower()
    rel = f"images/{bench}/{hashlib.sha1(data).hexdigest()[:16]}.{'jpg' if fmt == 'jpeg' else fmt}"
    p = os.path.join(out, rel)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    return rel


def copy_image(out, bench, src, name) -> str:
    rel = f"images/{bench}/{name}"
    p = os.path.join(out, rel)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(src, p)
    return rel


def strip_letter(opt: str) -> str:
    """'(A) text', 'A. text', 'A: text' -> 'text'."""
    return re.sub(r"^\s*\(?[A-K]\s*[\.\):]\s*", "", opt).strip()


def letter_split(text: str) -> list:
    """'A. x\\nB. y', 'A x\\nB y' or 'A. x B. y' -> ['x', 'y']: the markers A, B, C, ... found in order."""
    s, pos, i = text.strip(), [], 0
    for L in "ABCDEFGHIJK":
        m = re.compile(rf"(?:^|\s){L}(?:[\.:\)]|\s)\s*").search(s, i)
        if not m:
            break
        pos.append((m.start(), m.end()))
        i = m.end()
    return [s[e:(pos[k + 1][0] if k + 1 < len(pos) else len(s))].strip() for k, (_, e) in enumerate(pos)]


def mmstar_split(text: str) -> tuple:
    """MMStar question text -> (question, options). Two layouts: '...\\nOptions: A: x, B: y, C: z, D: w' and MathVista's
    'Hint: ...\\nQuestion: ...\\nChoices:\\n(A) x\\n(B) y'. Trailing "nan" options are dropped (an inner one stays, so the
    letters keep their meaning)."""
    if "Options:" in text:
        q, opts = text.rsplit("Options:", 1)
        marks = list(re.finditer(r"(?:^|,\s*)([A-F]):\s", opts.strip()))
        s = opts.strip()
        options = [s[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(s))].strip() for i, m in enumerate(marks)]
    else:                                   # MathVista style; "Choices:" is sometimes missing
        m = re.search(r"Question:\s*(.*)$", text, re.S)
        qlines, options = [], []
        for line in (m.group(1) if m else text).split("\n"):
            if re.match(r"^\s*\([A-F]\)", line):
                options.append(strip_letter(line))
            elif not options and line.strip() != "Choices:":
                qlines.append(line)
        q = "\n".join(qlines)
    while options and options[-1].lower() == "nan":
        options.pop()
    return q.strip(), options


def main(raw, out):
    items = []
    # MME-RealWorld-Lite
    d = os.path.join(raw, "MME-RealWorld-Lite", "data")
    for i, r in enumerate(json.load(open(os.path.join(d, "MME-RealWorld-Lite.json")))):   # its Question_id repeats
        ch = r["Answer choices"]
        ch = ast.literal_eval(ch) if isinstance(ch, str) else ch
        rel = copy_image(out, "mmerw_lite", os.path.join(d, "imgs", r["Image"]), r["Image"])
        items.append(item("mmerw_lite", i, r["Text"].strip(), [strip_letter(x) for x in ch], r["Ground truth"].strip(), rel,
                          out, f"{r['Task']}/{r['Subtask']}/{r['Category']}") | {"src_id": r["Question_id"]})
    # TreeBench
    with open(os.path.join(raw, "TreeBench", "TreeBench.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rel = save_bytes(out, "treebench", base64.b64decode(r["image"]))
            # "multi-choice options" holds them all ('A. x\nB. y'); 69 rows leave the A-K columns empty
            opts = letter_split(r.get("multi-choice options") or "") or \
                [r[k].strip() for k in "ABCDEFGHIJK" if (r.get(k) or "").strip()]
            items.append(item("treebench", r["index"], r["question"].strip(), opts, r["answer"].strip(), rel, out, r["category"]))
    # O3-Bench
    o3 = os.path.join(raw, "O3-Bench", "test")
    for i, line in enumerate(open(os.path.join(o3, "metadata.jsonl"))):
        r = json.loads(line)
        rel = copy_image(out, "o3bench", os.path.join(o3, r["file_name"]), r["file_name"].replace("images/", "", 1))
        opts = [strip_letter(x) for x in r["options"].split("\n") if x.strip()]
        items.append(item("o3bench", i, r["question"].strip(), opts, r["answer"].strip(), rel, out, r["subset"]))
    # MMStar
    for r in pq.read_table(os.path.join(raw, "MMStar", "mmstar.parquet")).to_pylist():
        q, opts = mmstar_split(r["question"])
        rel = save_bytes(out, "mmstar", r["image"])
        items.append(item("mmstar", r["index"], q, opts, r["answer"].strip(), rel, out, f"{r['category']}/{r['l2_category']}"))
    # HallusionBench (questions with an image)
    for fn in sorted(glob.glob(os.path.join(raw, "HallusionBench", "data", "image-*.parquet"))):
        for r in pq.read_table(fn).to_pylist():
            rel = save_bytes(out, "hallusion", r["image"]["bytes"])
            idx = f"{r['category']}_{r['subcategory']}_{r['set_id']}_{r['figure_id']}_{r['question_id']}"
            items.append(item("hallusion", idx, r["question"].strip(), YES_NO, "A" if str(r["gt_answer"]) == "1" else "B",
                              rel, out, f"{r['category']}/{r['subcategory']}"))
    qids = collections.Counter(it["qid"] for it in items)
    assert max(qids.values()) == 1, [q for q, n in qids.items() if n > 1][:5]
    bad = [it["qid"] for it in items if len(it["options"]) < 2 or ord(it["gold"]["o"]) - 65 >= len(it["options"])]
    assert not bad, bad[:5]
    with open(os.path.join(out, "items_gen.jsonl"), "w") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    c = collections.Counter(it["bench"] for it in items)
    imgs = collections.Counter(it["bench"] for it in {it["image"]["o"]: it for it in items}.values())
    print("items", dict(c), "| distinct images", dict(imgs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
