"""Metrics that need no verl: the step metrics of training, answer parsing, episodes, validation metrics and
arguments."""
import pytest

from vira.eval.args import EvalArguments, extra_eval_rows, parse_eval_args
from vira.eval.metrics import budget_tag, compute_metrics, episode_from_rollout, episode_from_row
from vira.eval.parsing import parse_letter, parse_strict
from vira.train.metrics import parse_key, rollout_rows, step_metrics

OPTIONS = ["red", "blue", "green", "white"]


def _info(acc, version, n_versions, x=1.0, zoom=1, cov=1.0, **kw):
    d = {"acc": float(acc), "answered": 1.0, "cov_max": cov, "cov_last": cov, "n_calls": float(zoom),
         "n_ok_calls": float(zoom), "n_errors": 0.0, "x": x, "version": float(version), "n_versions": float(n_versions),
         "truncated": 0.0, "open": 0.0, "has_box": 1.0, "judge_failed": 0.0, "chose_o": 0.0}
    d.update(kw)
    return d


def test_rollout_rows_and_step_metrics():
    keys = [f"aaaa_{s}_0" for s in range(6)] + ["bbbb_0_0", "bbbb_1_0"]
    assert parse_key("aaaa_11_2") == ("aaaa", 11, 2)
    extras = [{"reward_extra_info": _info(acc, s % 3, 3, zoom=1 if s < 4 else 0, cov=1.0 if s < 2 else 0.0,
                                          chose_o=float(s == 4)), "vira": {"source": "contrast_object"}}
              for s, acc in enumerate([1, 1, 0, 1, 0, 0])]
    extras += [{"reward_extra_info": _info(1, 0, 1, x=8.0, open=1.0, has_box=0.0), "vira": {"source": "detail"}},
               {"reward_extra_info": _info(0, 0, 1, x=8.0, open=1.0, has_box=0.0, judge_failed=1.0), "vira": {"source": "detail"}}]
    assert rollout_rows(keys, extras) == [("aaaa", 0), ("aaaa", 1), ("aaaa", 2), ("aaaa", 0), ("aaaa", 1), ("aaaa", 2),
                                          ("bbbb", 0), ("bbbb", 0)]
    m = step_metrics(extras)
    assert m["train/n"] == 8 and m["train/acc"] == 0.5 and m["train/zoom_rate"] == 0.75
    assert m["train/acc_O"] == 1.0 and m["train/acc_A"] == 0.5 and m["train/acc_B"] == 0.0
    assert m["train/hit_last"] == 0.5 and m["train/n_hit_last"] == 4                 # zoomed rollouts with a box
    assert m["train/acc_when_hit"] == 1.0 and m["train/chose_o_when_wrong"] == pytest.approx(1 / 3)
    assert m["train/acc_src_contrast_object"] == 0.5 and m["train/acc_src_detail"] == 0.5
    assert m["train/judge_failed"] == 0.5 and m["train_x/acc_1-2"] == 0.5 and m["train_x/n_8+"] == 2
    assert step_metrics(extras, prefix="train10")["train10/acc"] == 0.5 and step_metrics([]) == {}


def test_strict_and_lenient_reading():
    assert parse_strict("<answer>B</answer>") == "B" and parse_strict("The correct answer is B.") == ""
    assert parse_letter("The correct answer is B.", OPTIONS) == "B"
    assert parse_letter("<answer>blue</answer>", OPTIONS) == "B" and parse_letter("A or B", OPTIONS) == ""
    assert parse_letter("Looking at it closely...\n\nC.", OPTIONS) == "C" and parse_letter("<answer>E</answer>", OPTIONS) == ""


def test_budget_tag():
    assert [budget_tag(x) for x in (0.5, 1.0, 4, "2", 12.845056, 16.78, "max")] == \
        ["0.5mp", "1mp", "4mp", "2mp", "maxmp", "maxmp", "maxmp"]


def _extra(qid, version, x, reply, gold, golds, n_ok=1, cov_last=0.9, cov_max=0.9, source="contrast_object"):
    return {"vira": {"qid": qid, "version": version, "x": x, "gold": gold, "golds": golds, "options": OPTIONS,
                     "source": source, "last_reply": reply, "final_text": reply[-600:],
                     "calls": [{"bbox": [1, 2, 3, 4]}, {"error": "x"}][: n_ok * 2]},
            "reward_extra_info": {"n_ok_calls": float(n_ok), "cov_last": cov_last, "cov_max": cov_max, "truncated": 0.0}}


def test_episode_from_rollout():
    golds = {"O": "A", "A": "B", "B": "C"}
    e = episode_from_rollout(_extra("q1", "A", 1.0, "It is blue, so B.", "B", golds), seq_tokens=1234)
    assert e["budget"] == "1mp" and e["version"] == "A" and e["boxes"] == [[1, 2, 3, 4]]
    assert not e["strict"] and e["lenient"] and not e["answered"] and e["letter"] == "B"
    assert e["hit"] and e["gold_o"] == "A" and e["seq_tokens"] == 1234 and e["tokens"] is None
    e = episode_from_rollout(_extra("q1", "B", 4.0, "<answer>A</answer>", "C", golds, n_ok=0), seq_tokens=10)
    assert e["answered"] and not e["strict"] and e["hit"] is None and e["letter"] == "A"
    x = _extra("d1", "O", 0.5, "<answer>a red door</answer>", "red door", {"O": "red door"}, source="detail")
    x["reward_extra_info"].update(open=1.0, acc=1.0, answered=1.0)
    x["vira"]["answer"] = "a red door"
    e = episode_from_rollout(x)
    assert e["open"] and e["strict"] and e["lenient"] and e["answered"] and e["source"] == "detail"


def _set(qid, x, right, zoom=True):
    golds = {"O": "A", "A": "B", "B": "C"}
    out = []
    for v in "OAB":
        ans = golds[v] if right[v] else golds["O"] if v != "O" else "D"
        out.append(episode_from_rollout(_extra(qid, v, x, f"<answer>{ans}</answer>", golds[v], golds,
                                               n_ok=1 if zoom else 0), seq_tokens=100))
    return out


def test_compute_metrics_names_and_values():
    eps = (_set("q1", 1.0, {"O": True, "A": True, "B": True})
           + _set("q2", 1.0, {"O": True, "A": False, "B": True})
           + _set("q1", 4.0, {"O": True, "A": True, "B": True}, zoom=False))
    m = compute_metrics(eps, "dev")
    assert m["dev/set_acc_strict_1mp"] == 50.0 and m["dev/set_acc_lenient_1mp"] == 50.0
    assert m["dev/acc_1mp"] == pytest.approx(500 / 6) and m["dev/acc_A_1mp"] == 50.0
    assert m["dev/chose_o_when_wrong_1mp"] == 100.0          # the wrong answer on A is the original's answer
    assert m["dev/n_questions_1mp"] == 2 and m["dev/zooms_4mp"] == 0 and "dev/hit_last_4mp" not in m
    assert m["dev/seq_tokens_1mp"] == 100 and "dev/tokens_1mp" not in m
    assert m["dev/acc_all_1mp"] == m["dev/acc_1mp"] and "dev/acc_macro_mean" not in m


def test_episode_from_row():
    items = {"q1": {"diff_bbox": [10, 10, 20, 20], "gold": {"o": "A"}}}
    row = {"qid": "q1", "version": "a", "budget": "4", "gold": "B", "correct": True, "pred": "B",
           "zooms": [{"bbox": [0, 0, 30, 30]}, {"error": "x"}], "replies": ["<tool_call>..</tool_call>", "<answer>B</answer>"],
           "prompt_tokens_sum": 900, "completion_tokens": 100}
    e = episode_from_row(row, items)
    assert e["budget"] == "4mp" and e["version"] == "A" and e["strict"] and e["hit"] and e["zooms"] == 1
    assert e["tokens"] == 1000 and e["gold_o"] == "A"
    e = episode_from_row(dict(row, budget="max", replies=["The answer is B."], zooms=[]))
    assert e["budget"] == "maxmp" and not e["strict"] and e["lenient"] and e["hit"] is None


def _ep(qid, b, ok, source, version="O"):
    return {"qid": qid, "version": version, "budget": b, "strict": ok, "lenient": ok, "open": source != "contrast_object",
            "source": source, "answered": True, "zooms": 1, "hit": True, "cov_max": 1.0, "letter": "A", "gold": "A",
            "gold_o": "A", "tokens": None, "seq_tokens": 10, "x": 0.5}


def test_compute_metrics_validation_groups():
    eps = ([_ep("t1", "0.5mp", ok, "contrast_object", v) for v, ok in zip("OAB", (True, True, False))]
           + [_ep("c1", "0.5mp", True, "detail"), _ep("c2", "0.5mp", False, "detail")]
           + [_ep("p1", "0.5mp", True, "multi_evidence_photo"), _ep("m1", "0.5mp", False, "math"),
              _ep("h1", "0.5mp", True, "chart"), _ep("h2", "0.5mp", True, "chart")])
    m = compute_metrics(eps, "dev")
    assert m["dev/acc_0.5mp"] == pytest.approx(200 / 3) and m["dev/n_questions_0.5mp"] == 1   # contrast sets only
    assert m["dev_detail/acc_0.5mp"] == 50.0
    assert m["dev_multi_evidence_photo/acc_0.5mp"] == 100.0 and m["dev_math/acc_0.5mp"] == 0.0
    assert m["dev_chart/acc_0.5mp"] == 100.0
    assert m["dev_other/acc_0.5mp"] == 75.0 and m["dev/acc_all_0.5mp"] == pytest.approx(600 / 9)
    assert m["dev/acc_macro_0.5mp"] == pytest.approx((200 / 3 + 50 + 75) / 3) == m["dev/acc_macro_mean"]
    part = compute_metrics(eps[:5], "dev")                     # no other single-image types: no macro metric
    assert "dev/acc_macro_mean" not in part and part["dev_detail/acc_0.5mp"] == 50.0


def test_eval_arguments():
    a = parse_eval_args(["--eval_dataset", "dev.jsonl", "--eval_output_dir", "/o", "--eval_steps", "25",
                         "--eval_on_start", "true", "--eval_budgets_mp", "1", "4"])
    assert a.eval_budgets_mp == (1.0, 4.0) and a.eval_on_start is True and a.enabled and a.eval_max_new_tokens == 4096
    ov = a.to_verl_overrides()
    assert "trainer.test_freq=25" in ov and "trainer.val_before_train=True" in ov
    assert "data.val_files=/o/eval_dataset.parquet" in ov and "actor_rollout_ref.rollout.val_kwargs.temperature=0" in ov
    assert "+vira.eval.output_dir=/o" in ov
    assert EvalArguments(eval_strategy="no").to_verl_overrides() == ["trainer.val_before_train=False", "trainer.test_freq=-1"]
    with pytest.raises(ValueError):
        EvalArguments(eval_strategy="epoch")
    assert parse_eval_args(["--eval_on_start", "false"]).eval_on_start is False


def test_extra_eval_parquet(tmp_path):
    from vira.data import write_parquet

    rows = [{"data_source": "vira_eval/chart", "agent_name": "vira_agent", "prompt": [{"role": "user", "content": "q"}],
             "reward_model": {"style": "rule", "ground_truth": '{"O": "A"}'},
             "extra_info": {"qid": f"p{i}", "eval": True, "x_fixed": b, "max_new_tokens_per_turn": None, "source": "chart",
                            "options": ["l", "r"], "variants": [{"name": "O", "path": "/x.jpg", "gold": "A"}]}}
            for i, b in enumerate((0.5, 4.0, 1.0))]
    write_parquet(rows, str(tmp_path / "d.parquet"))
    got = extra_eval_rows(str(tmp_path / "d.parquet"), (0.5, 4.0), 4096)
    assert [r["extra_info"]["qid"] for r in got] == ["p0", "p1"]                       # the 1 MP row is dropped
    assert all(r["extra_info"]["max_new_tokens_per_turn"] == 4096 for r in got)
    assert isinstance(got[0]["extra_info"]["options"], list)
