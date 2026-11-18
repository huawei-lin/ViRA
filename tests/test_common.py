"""vira.common: the pixel budget, sizes, boxes, answer parsing and the CrossRPO advantage."""
import math

import pytest

from vira import common as C


def test_budget_same_for_group_and_log_uniform():
    assert C.budget_mp("g1") == C.budget_mp("g1")
    xs = [C.budget_mp(f"g{i}", lo=0.5, hi=12.8) for i in range(20000)]
    assert min(xs) >= 0.5 and max(xs) <= 12.8
    edges = [0.5, 1, 2, 4, 8, 12.8]
    frac = [sum(a <= x < b for x in xs) / len(xs) for a, b in zip(edges, edges[1:])]
    width = [math.log(b / a) / math.log(12.8 / 0.5) for a, b in zip(edges, edges[1:])]
    for f, w in zip(frac, width):
        assert abs(f - w) < 0.015
    assert C.budget_mp("g1", seed=1) != C.budget_mp("g1", seed=2)


def test_budgets_of_the_evaluation():
    assert C.BUDGETS == {"0.5": 501760, "1": 1003520, "2": 2007040, "4": 4014080, "max": 16777216}


def test_resized_and_magnified_size():
    w, h = C.resized_size(8000, 6000, 1_000_000)
    assert w % 32 == 0 and h % 32 == 0 and w * h <= 1_000_000 and abs(w / h - 8000 / 6000) < 0.05
    # a 45 MP photo (8256 x 5504) under each budget of the evaluation
    want = {"0.5": (864, 576), "1": (1216, 800), "2": (1728, 1152), "4": (2432, 1632), "max": (4992, 3328)}
    assert {b: C.resized_size(8256, 5504, x) for b, x in C.BUDGETS.items()} == want
    assert C.resized_size(100, 100, 10 ** 6) == (256, 256)                       # at least MIN_PIXELS
    assert C.resized_size(100, 100, 10 ** 6, min_pixels=4096) == (96, 96)
    # a crop is enlarged to fill the budget, by at most 4 x per side; a crop above the budget is shrunk
    assert C.magnified_size(100, 100, 10 ** 6) == (384, 384)                     # 4 x, on the patch grid
    w, h = C.magnified_size(600, 400, 10 ** 6)
    assert w * h <= 10 ** 6 and 1.9 < w / 600 < 2.1
    assert C.magnified_size(4000, 3000, 10 ** 6) == C.resized_size(4000, 3000, 10 ** 6)


def test_boxes_and_coverage():
    assert C.view_to_orig([10, 20, 30, 40], (100, 100), (1000, 2000)) == [100, 400, 300, 800]
    box = [100, 100, 200, 200]
    assert C.coverage([0, 0, 1000, 1000], box) == 1.0
    assert C.coverage([150, 0, 1000, 1000], box) == 0.5
    assert C.coverage([300, 300, 400, 400], box) == 0.0


def test_parse_answer():
    t = "<think>look</think><tool_call>{}</tool_call> ... <answer> **B** </answer>"
    assert C.has_answer(t) and C.parse_answer(t) == "B"
    assert C.parse_answer("<answer>(C) red</answer>") == "C"
    assert C.parse_answer("<answer>A</answer> no, <answer>D. white</answer>") == "D"       # the last answer counts
    assert C.parse_answer("no answer here") == "" and C.parse_answer("<answer>blue</answer>") == ""


def test_crossrpo_advantages_worked_example():
    """One contrast set, 4 rollouts per version: O 3/4, A 2/4, B 1/4 correct, lambda = 0.5."""
    c = {"O": [1, 1, 1, 0], "A": [1, 1, 0, 0], "B": [1, 0, 0, 0]}
    a = C.crossrpo_advantages(["q"] * 12, [v for v in "OAB" for _ in range(4)], [x for v in "OAB" for x in c[v]], 0.5)
    A = dict(zip([f"{v}{i + 1}" for v in "OAB" for i in range(4)], a))
    w = {"O": 0.5 * 0.25, "A": 0.75 * 0.25, "B": 0.75 * 0.5}                  # the product of the other versions' p
    assert A["O1"] == pytest.approx((1 + 0.5 * w["O"]) * (1 - 2 / 3))
    assert A["O4"] == pytest.approx((1 + 0.5 * w["O"]) * (0 - 1))
    assert A["A1"] == pytest.approx((1 + 0.5 * w["A"]) * (1 - 1 / 3))
    assert A["B1"] == pytest.approx((1 + 0.5 * w["B"]) * 1.0)                 # the least accurate version: largest weight
    assert A["B2"] == pytest.approx((1 + 0.5 * w["B"]) * (0 - 1 / 3))
    for v in "OAB":                                                            # leave-one-out terms sum to 0 per version
        assert abs(sum(A[f"{v}{i + 1}"] for i in range(4))) < 1e-9


def test_crossrpo_advantages_edge_cases():
    # a guess: O always correct, A and B always wrong -> every advantage is 0
    assert C.crossrpo_advantages(["q"] * 12, [v for v in "OAB" for _ in range(4)], [1] * 4 + [0] * 8) == [0.0] * 12
    # a set without an original (A and B only)
    a = C.crossrpo_advantages(["g"] * 4, ["A", "A", "B", "B"], [1, 0, 1, 1], 0.5)
    assert a[0] == pytest.approx(1 + 0.5 * 1.0) and a[1] == pytest.approx(-(1 + 0.5 * 1.0)) and a[2] == a[3] == 0
    # a single image: the leave-one-out advantage alone; a version with one rollout: 0
    assert C.crossrpo_advantages(["g"] * 3, ["O"] * 3, [1, 0, 0]) == [1.0, -0.5, -0.5]
    assert C.crossrpo_advantages(["g", "h"], ["A", "A"], [1, 0]) == [0.0, 0.0]


def test_lambda0_trains_each_version_separately():
    groups = ["q"] * 8 + ["s"] * 3
    versions = ["O", "O", "A", "A", "A", "A", "B", "B", "O", "O", "O"]
    c = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    a = C.crossrpo_advantages(groups, versions, c, lam=0.0)
    assert a[:2] == [1.0, -1.0] and a[2:6] == [1 - 1 / 3, 1 - 1 / 3, -2 / 3, -2 / 3] and a[6:8] == [0.0, 0.0]
    assert a[8:] == [1.0, -0.5, -0.5]
    assert C.crossrpo_advantages(groups, versions, c[:6] + [0, 0] + c[8:], lam=0.0)[:6] == a[:6]   # O, A ignore B


def test_crossrpo_unbiased_monte_carlo():
    """On a toy policy (version v is answered correctly with p_v = sigmoid(theta_v)), the CrossRPO estimate
    (1/n) sum_i A_i * dlog pi_i / dtheta_v matches dJ/dtheta_v of J = sum_v p_v + lam prod_v p_v."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    p, n, lam, G = np.array([0.8, 0.5, 0.3]), 4, 0.5, 40000
    c = (rng.random((G, 3, n)) < p[None, :, None]).astype(float)
    adv = np.array([C.crossrpo_advantages(["q"] * 12, [v for v in range(3) for _ in range(n)], g.ravel().tolist(), lam)
                    for g in c]).reshape(G, 3, n)
    est = (adv * (c - p[None, :, None])).mean(-1).mean(0)                    # score of a Bernoulli logit: c - p
    others = np.array([p[1] * p[2], p[0] * p[2], p[0] * p[1]])
    assert np.allclose(est, p * (1 - p) * (1 + lam * others), rtol=0.03)
