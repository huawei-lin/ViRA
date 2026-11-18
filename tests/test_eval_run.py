"""The offline evaluation (vira.eval.run, vira.eval.summarize) with a scripted model."""
import json

import pytest
from PIL import Image

from vira import common as C
from vira.eval import run as R
from vira.eval import summarize as S
from vira.eval import vlm as V

CALL = '<think>look</think> <tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": %s}}'
USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "sec": 0.0, "finish": "stop"}


def test_view_is_cached(tmp_path):
    src = tmp_path / "photo.png"
    Image.new("RGB", (2000, 1500), (10, 20, 30)).save(src)
    first = V.view(str(src), C.BUDGETS["0.5"], str(tmp_path / "cache"))
    assert first[1:] == ((800, 608), (2000, 1500)) and Image.open(first[0]).size == (800, 608)
    assert V.view(str(src), C.BUDGETS["0.5"], str(tmp_path / "cache")) == first
    small = tmp_path / "small.png"
    Image.new("RGB", (100, 90)).save(small)
    assert V.view(str(small), C.BUDGETS["max"], str(tmp_path / "cache"))[1] == (96, 96)   # only rounded to the grid


class FakeCaller:
    """Answers each version with its correct letter after one zoom; version b of q2 answers without the tag."""

    seen = []

    def __init__(self, urls, model):
        self.urls, self.model = urls, model

    def __call__(self, key, messages):
        FakeCaller.seen.append((key, len(messages)))
        gold = {"q1_o": "A", "q1_a": "B", "q1_b": "C", "q2_o": "B", "q2_a": "C", "q2_b": "A", "x1_o": "A"}[key]
        if len(messages) == 2:
            return CALL % "[450, 450, 550, 550]", dict(USAGE)
        return ("The answer is %s." if key == "q2_b" else "<answer>%s</answer>") % gold, dict(USAGE)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "Caller", FakeCaller)
    FakeCaller.seen = []
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    items = []
    for q, golds in (("q1", "ABC"), ("q2", "BCA")):
        for v in "oab":
            Image.new("RGB", (3000, 2000), "white").save(root / "images" / f"{q}_{v}.png")
        items.append({"qid": q, "pair_id": q, "question": "What?", "options": ["w", "x", "y"],
                      "gold": dict(zip("oab", golds)), "image": {v: f"images/{q}_{v}.png" for v in "oab"},
                      "image_size": [3000, 2000], "diff_bbox": [1450, 950, 1550, 1050]})
    with open(root / "items.jsonl", "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)
    return root


def _run(root, out, budget):
    R.main(["--model", "ck", "--budget", budget, "--items", str(root / "items.jsonl"), "--data-root", str(root),
            "--urls", "http://a/v1,http://b/v1", "--out", str(out), "--cache", str(root / "cache"), "--workers", "4"])
    with open(out) as f:
        return [json.loads(line) for line in f]


def test_run_rows_and_resume(bench, tmp_path):
    out = tmp_path / "eval" / "ViRA" / "bench_1mp.jsonl"
    rows = _run(bench, out, "1")
    assert len(rows) == 6 and {(r["qid"], r["version"]) for r in rows} == {(q, v) for q in ("q1", "q2") for v in "oab"}
    r = next(r for r in rows if (r["qid"], r["version"]) == ("q1", "a"))
    assert r["budget"] == "1" and r["correct"] and r["pred"] == "B" and r["status"] == "ok" and r["n_zoom"] == 1
    assert r["first_view"] == {"size": [1216, 800], "orig_size": [3000, 2000]} and r["calls"] == 2
    z = r["zooms"][0]
    assert z["bbox"] == [1350, 900, 1650, 1100] and z["source"] == 0 and z["fed"][0] * z["fed"][1] <= C.BUDGETS["1"]
    assert r["pixels_encoded"] == 1216 * 800 + z["fed"][0] * z["fed"][1]
    assert r["visual_tokens"] == (1216 // 32) * (800 // 32) + (z["fed"][0] // 32) * (z["fed"][1] // 32)
    assert r["prompt_tokens_sum"] == 200 and r["completion_tokens"] == 20 and len(r["replies"]) == 2
    assert Image.open(tmp_path / "eval" / "ViRA" / "crops" / "1mp" / "q1_a_1.png").size == tuple(z["fed"])
    n = len(FakeCaller.seen)
    assert len(_run(bench, out, "1")) == 6 and len(FakeCaller.seen) == n       # finished rows are not run again


def test_summarize(bench, tmp_path, capsys):
    d = tmp_path / "eval"
    _run(bench, d / "ViRA" / "bench_1mp.jsonl", "1")
    _run(bench, d / "ViRA" / "bench_maxmp.jsonl", "max")
    (d / "items_bench.jsonl").write_text((bench / "items.jsonl").read_text())
    S.main(str(d))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ViRA")]
    assert len(lines) == 2
    cells = [c.strip() for c in lines[0].split("|")]
    # q1: all three versions correct; q2: version b has no <answer> (lenient reading correct, strict wrong)
    assert cells[1:8] == ["ViRA", "1 MP", "50.0", "100.0", "83.3", "100.0 / 100.0 / 50.0", "83.3"]
    assert cells[8] == "100.0" and cells[9] == "83.3" and cells[10] == "1.00" and cells[12] == "2"
    assert [c.strip() for c in lines[1].split("|")][2] == "max"
