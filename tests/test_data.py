"""vira.data (rows of the verl parquet files) and vira.hf (Hub rows -> items and training rows), without the Hub."""
import json

from vira import hf
from vira.data import build_eval

OPTIONS = ["red", "blue", "green", "white"]


def _set(qid="s1", k=3, **kw):
    r = {"qid": qid, "pair_id": "p1", "type": "contrast_object", "question": "Q?", "options": ["w", "x", "y", "z"],
         "answer_type": "letter", "answer_o": "A", "answer_a": "B", "answer_b": "C",
         "image_o": "images/g/g_O.jpg", "image_a": "images/g/p1_A.jpg", "image_b": "images/g/p1_B.jpg",
         "image_size": [6000, 4000], "diff_bbox": [10, 20, 60, 80], "evidence_boxes": [[10.0, 20.0, 60.0, 80.0]], "meta": "{}"}
    if k == 2:
        r.update(answer_o=None, image_o=None)
    return dict(r, **kw)


def _single(qid="q1", type="multi_evidence_photo", answer_type="open", **kw):
    r = {"qid": qid, "pair_id": None, "type": type, "question": "What?", "options": [], "answer_type": answer_type,
         "answer_o": "gold", "answer_a": None, "answer_b": None, "image_o": "images/h/h_O.jpg", "image_a": None,
         "image_b": None, "image_size": [3000, 2000], "diff_bbox": None,
         "evidence_boxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], "meta": json.dumps({"evidence_bbox": [5, 6, 7, 8]})}
    return dict(r, **kw)


def test_build_eval_rows():
    it = {"qid": "q1", "question": "What color?", "options": OPTIONS, "versions": ["o", "a", "b"],
          "gold": {"o": "A", "a": "B", "b": "C"}, "images": {k: f"images/{k}.jpg" for k in "oab"},
          "image_size": [6000, 4000], "diff_bbox": [1, 2, 3, 4], "source": "contrast_object"}
    rows = build_eval([it], "/data", (1.0, 4.0), max_new_tokens_per_turn=4096)
    ex = [r["extra_info"] for r in rows]
    assert [(e["x_fixed"], e["variants"][0]["name"]) for e in ex] == [(1.0, "O"), (1.0, "A"), (1.0, "B"),
                                                                      (4.0, "O"), (4.0, "A"), (4.0, "B")]
    assert all(e["eval"] and e["max_new_tokens_per_turn"] == 4096 and e["golds"] == {"O": "A", "A": "B", "B": "C"}
               and len(e["variants"]) == 1 and e["qid"] == "q1" for e in ex)
    assert ex[1]["variants"][0]["path"] == "/data/images/a.jpg" and rows[0]["data_source"] == "vira_eval/contrast_object"
    two = build_eval([dict(it, versions=["a", "b"])], "/data", (1.0,))
    assert [r["extra_info"]["variants"][0]["name"] for r in two] == ["A", "B"]


def test_bench_item():
    r = {"qid": "q", "pair_id": "p", "task": "multi", "kind": "symbol", "question": "Q?", "options": ["a", "b", "c", "d"],
         "answer_o": "B", "answer_a": "D", "answer_b": "C", "image_o": "images/x_O.jpg", "image_a": "images/p_A.jpg",
         "image_b": "images/p_B.jpg", "image_size": [7296, 5472], "bbox": [4570, 2174, 4624, 2237]}
    it = hf.bench_item(r)
    assert it["gold"] == {"o": "B", "a": "D", "b": "C"} and it["image"]["a"] == "images/p_A.jpg"
    assert it["pair_id"] == "q" and it["set"] == "p" and it["subset"] == "multi"
    assert it["diff_bbox"] == it["core_bbox"] == [4570.0, 2174.0, 4624.0, 2237.0]
    real = hf.bench_item(dict(r, bbox=[10, 10, 20, 20], bbox_a=[5, 12, 18, 25], bbox_b=[11, 8, 22, 19]))
    assert real["core_bbox"] == [10.0, 10.0, 20.0, 20.0] and real["diff_bbox"] == [5, 8, 22, 25]


def test_train_rows_follow_the_order():
    rows = [_set("s1"), _set("s2", k=2), _single("q1"), _single("q2", "detail"),
            _single("q3", "multi_evidence_map", "letter", options=["l", "r"], answer_o="B"), _single("unused")]
    out = hf.train_rows(rows, ["q3", "s2", "q1", "s1", "q2"], "/root")
    ex = [r["extra_info"] for r in out]
    assert [e["qid"] for e in ex] == ["q3", "s2", "q1", "s1", "q2"] and [e["index"] for e in ex] == list(range(5))
    assert [len(e["variants"]) for e in ex] == [1, 2, 1, 3, 1]
    assert [e["source"] for e in ex] == ["multi_evidence_map", "contrast_object", "multi_evidence_photo",
                                         "contrast_object", "detail"]
    assert [e["open"] for e in ex] == [False, False, True, False, True]
    assert [v["name"] for v in ex[1]["variants"]] == ["A", "B"] and ex[1]["variants"][0]["path"] == "/root/images/g/p1_A.jpg"
    assert json.loads(out[3]["reward_model"]["ground_truth"]) == {"O": "A", "A": "B", "B": "C"}
    assert ex[3]["evidence_bbox"] == [10.0, 20.0, 60.0, 80.0] and ex[2]["evidence_bbox"] == [5.0, 6.0, 7.0, 8.0]
    assert out[0]["data_source"] == "vira/multi_evidence_map" and out[0]["agent_name"] == "vira_agent"
    assert len({tuple(sorted(e)) for e in ex}) == 1                    # one schema for the parquet


def test_dev_files():
    rows = [_set("d1"), _single("d2", "detail"), _single("d3", "chart")]
    items = hf.dev_items(rows)
    assert len(items) == 1 and items[0]["versions"] == ["o", "a", "b"] and items[0]["images"]["b"] == "images/g/p1_B.jpg"
    assert len(build_eval(items, "/root", (0.5,))) == 3
    single = hf.dev_single_rows(rows, "/root", budgets_mp=(0.5, 4.0))
    assert len(single) == 4 and {r["data_source"] for r in single} == {"vira_eval/detail", "vira_eval/chart"}
    ex = single[1]["extra_info"]
    assert ex["eval"] is True and ex["x_fixed"] == 4.0 and ex["golds"] == {"O": "gold"} and ex["open"] is True
    assert hf.image_paths(rows) == ["images/g/g_O.jpg", "images/g/p1_A.jpg", "images/g/p1_B.jpg", "images/h/h_O.jpg"]
