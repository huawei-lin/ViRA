"""Offline evaluation of a model under one pixel budget, on contrast sets (ViRA-Bench, ViRA-Bench-Real: the versions
O/A/B of each set) or single-image items (the general benchmarks).

    python -m vira.eval.run --model NAME --budget B --items ITEMS --data-root D --urls U1,U2 --out OUT \\
        [--cache DIR] [--workers N] [--limit N]

NAME is the model name on the servers; ``scripts/serve.sh`` starts them (one OpenAI-compatible vLLM server per GPU,
greedy, ``--allowed-local-media-path`` covering --data-root and --cache). B is the pixel budget: every image the
model sees (the first view and every crop) has at most 0.5 / 1 / 2 / 4 MP, or the largest image of the model (max:
16.8 MP); see ``common.BUDGETS``. Every episode is one rollout of ``vira.agent``.

Each row: the answer (letter, lenient ``parsing.parse_letter``), correctness, every zoom (the box as the model wrote
it, the box in pixels of the photo, the size shown) and the cost: pixels encoded, visual tokens, prompt / completion
tokens of every call, wall time. The crops are saved next to OUT (crops/<budget>/). Resumable: rows already in OUT
are skipped (status "error" is retried).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from vira import agent as A
from vira import common as C
from vira import images as I
from vira.eval import vlm as V
from vira.eval.metrics import budget_tag
from vira.eval.parsing import answer_text, parse_letter, strip_think

VIRA_HOME = os.environ.get("VIRA_HOME", os.path.expanduser("~/vira"))


def run_one(a, call, item, v) -> dict:
    """One (item, version) under the budget -> a result row."""
    key = f"{item['qid']}_{v}"
    row = {"qid": item["qid"], "pair_id": item["pair_id"], "version": v, "budget": a.budget, "gold": item["gold"][v]}
    usage = []

    def chat(messages):
        text, u = call(key, messages)
        usage.append(u)
        return text

    try:
        photo = os.path.join(a.data_root, item["image"][v])
        vfile, vsize, osize = V.view(photo, a.pixels, a.cache)
        out = A.rollout(chat, item["question"], item["options"], photo, osize, a.pixels, V.image_part(vfile), vsize,
                        lambda im, k: V.image_part(V.save_image(im, f"{a.crop_dir}/{key}_{k}.png")))
    except Exception as e:  # noqa: BLE001 - recorded; "error" rows are retried on the next run
        msg = f"{type(e).__name__}: {e}"
        row.update(status="overflow" if "maximum context length" in msg or "too long" in msg else "error",
                   error=msg[:500], pred="", correct=False)
        return row
    text = out["replies"][-1]
    pred = parse_letter(text, item["options"])
    shown = [vsize] + [z["fed"] for z in out["zooms"] if z.get("bbox")]
    row.update(pred=pred, correct=pred == item["gold"][v], status=out["status"],
               answer_raw=(answer_text(text) or strip_think(text))[-300:],
               first_view={"size": list(vsize), "orig_size": list(osize)}, zooms=out["zooms"], n_zoom=len(shown) - 1,
               pixels_encoded=sum(w * h for w, h in shown),
               visual_tokens=sum((w // C.PATCH) * (h // C.PATCH) for w, h in shown), calls=len(usage),
               prompt_tokens_last=usage[-1]["prompt_tokens"], prompt_tokens_sum=sum(u["prompt_tokens"] for u in usage),
               completion_tokens=sum(u["completion_tokens"] for u in usage),
               sec=round(sum(u["sec"] for u in usage), 3), finish=[u["finish"] for u in usage], replies=out["replies"])
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vira.eval.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="model name on the servers")
    ap.add_argument("--budget", required=True, choices=list(C.BUDGETS), help="pixel budget in MP")
    ap.add_argument("--items", required=True, help="items jsonl (ViRA-Bench items.jsonl, or items_ext.jsonl / "
                                                   "items_gen.jsonl of the general benchmarks)")
    ap.add_argument("--data-root", required=True, help="directory the items' image paths are relative to")
    ap.add_argument("--urls", required=True, help="comma-separated OpenAI-compatible base URLs (scripts/serve.sh urls)")
    ap.add_argument("--out", required=True, help="results jsonl, appended")
    ap.add_argument("--cache", default=os.path.join(VIRA_HOME, "eval", "cache"), help="resized first views")
    ap.add_argument("--workers", type=int, default=32, help="episodes in flight")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items")
    a = ap.parse_args(argv)
    a.pixels = C.BUDGETS[a.budget]
    out_dir = os.path.dirname(os.path.abspath(a.out))
    a.crop_dir = f"{out_dir}/crops/{budget_tag(a.budget)}"
    os.makedirs(out_dir, exist_ok=True)
    I.set_original_cache(int(os.environ.get("VIRA_ORIG_CACHE", "256")))    # speed / RAM only, not results
    with open(a.items) as f:
        items = [json.loads(line) for line in f]
    if a.limit:
        items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["qid"], r["version"]) for r in map(json.loads, f) if r.get("status") != "error"}
    call = V.Caller(a.urls.split(","), a.model)
    jobs = [(it, v) for it in items for v in "abo"            # a single-image item has only "o"
            if v in it["image"] and (it["qid"], v) not in done]
    random.Random(0).shuffle(jobs)                            # mix versions and photo sizes across the servers
    print(a.model, a.budget, len(items), "items,", len(jobs), "episodes to do", flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for fu in as_completed([ex.submit(run_one, a, call, it, v) for it, v in jobs]):
            f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(a.out) as f:
        rows = [json.loads(line) for line in f]
    acc = {v: [r["correct"] for r in rows if r["version"] == v] for v in "abo"}
    print(a.model, a.budget, {v: f"{sum(x)}/{len(x)}" for v, x in acc.items()},
          "status", {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})})


if __name__ == "__main__":
    main()
