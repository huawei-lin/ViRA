"""vira.train.crossrpo: the CrossRPO advantage inside verl (the final output of each rollout is scored, its advantage
goes to every token, batch scale)."""
import math

import pytest

torch = pytest.importorskip("torch")

from vira import common as C  # noqa: E402
from vira.train.crossrpo import make_crossrpo_fn  # noqa: E402


class _Data:
    def __init__(self, mask, rewards):
        tlr = torch.zeros_like(mask)
        for i, r in enumerate(rewards):
            n = int(mask[i].sum())
            if n:
                tlr[i, n - 1] = r                       # verl puts the reward of a rollout on its last token
        self.batch = {"response_mask": mask, "token_level_rewards": tlr}


def _mask(lengths, L):
    m = torch.zeros(len(lengths), L)
    for i, n in enumerate(lengths):
        m[i, :n] = 1
    return m


def test_matches_common_and_scales_by_the_batch_rms():
    # one contrast set (uid g): versions 0, 1, 2 with 2 rollouts each; one single image (uid s) with 3 rollouts
    keys = [f"g_{s}_0" for s in range(6)] + [f"s_{s}_0" for s in range(3)]
    rows = [("g", v) for v in (0, 0, 1, 1, 2, 2)] + [("s", 0)] * 3
    c = [1, 0, 1, 1, 0, 1, 1, 0, 0]
    lengths = [5, 3, 4, 5, 0, 2, 3, 3, 1]                              # row 4: a rollout without response tokens
    metrics = {}
    out = make_crossrpo_fn(rows, 0.5, metrics)(_Data(_mask(lengths, 5), c), keys)
    seen = [x if n else 0 for x, n in zip(c, lengths)]
    want = C.crossrpo_advantages([r[0] for r in rows], [r[1] for r in rows], seen, 0.5)
    rms = math.sqrt(sum(a * a for a, n in zip(want, lengths) if n) / sum(n > 0 for n in lengths))
    assert metrics["train/crossrpo_rms"] == pytest.approx(rms)
    assert metrics["train/crossrpo_nonzero"] == pytest.approx(sum(abs(a) > 0 for a, n in zip(want, lengths) if n) / 8)
    for i, n in enumerate(lengths):
        assert torch.allclose(out.batch["advantages"][i], want[i] / (rms + 1e-6) * _mask(lengths, 5)[i], atol=1e-6)
    assert torch.equal(out.batch["returns"], out.batch["advantages"])


def test_uses_the_final_output_of_each_rollout():
    # rollout 0 has two outputs (index 0 and 1): the final one is scored, both rows get its advantage
    keys = ["g_0_0", "g_0_1", "g_1_0", "g_2_0"]
    rows = [("g", 0)] * 4
    data = _Data(torch.ones(4, 2), [0, 1, 0, 0])
    out = make_crossrpo_fn(rows, 0.5)(data, keys)
    want = C.crossrpo_advantages(["g"] * 3, [0, 0, 0], [1, 0, 0], 0.5)
    rms = math.sqrt(sum(a * a for a in want) / 3)
    for row, a in ((0, want[0]), (1, want[0]), (2, want[1]), (3, want[2])):
        assert torch.allclose(out.batch["advantages"][row], torch.full((2,), a / (rms + 1e-6)), atol=1e-6)
