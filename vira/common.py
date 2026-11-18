"""The numbers of ViRA and of CrossRPO, framework-free (standard library only). Training (vira.train), the protocol
(vira.protocol) and the evaluation (vira.eval) import them, so they are the same everywhere.

  budget_mp                        pixel budget x of a contrast set (log-uniform, the same for all its versions)
  resized_size / magnified_size    size of the first view and of a zoom crop under the pixel budget
  view_to_orig                     box in an image that the model saw -> pixels of the photo
  coverage                         share of the object's box inside a crop (a logged diagnostic)
  crossrpo_advantages              the CrossRPO advantage of each rollout, before the batch scale
  parse_answer / has_answer        option letter inside the last <answer></answer>
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

CROSSRPO_LAMBDA = 0.5          # lambda: weight of the joint term of the objective
MAGNIFY = 4.0                  # a zoom crop is enlarged to fill the budget x, by at most this factor per side
HIT_COVERAGE = 0.5             # a crop covers the object when it contains at least half of the object's box
# images of Qwen3.5 and Qwen3.8, the base models of ViRA-9B and ViRA-27B
PATCH = 32                     # pixels per side of one visual token (16-px patches, merged 2 x 2)
MIN_PIXELS = 65536             # smallest image
MAX_PIXELS = 16384 * 32 * 32   # largest image: 16.78 MP, the largest pixel budget
# pixel budgets of the evaluation: 0.5, 1, 2 and 4 MP counted in 28 x 28-pixel units (1 MP = 1,280 units =
# 1,003,520 pixels), and the largest image
BUDGETS = {"0.5": 640 * 28 * 28, "1": 1280 * 28 * 28, "2": 2560 * 28 * 28, "4": 5120 * 28 * 28, "max": MAX_PIXELS}


# ------------------------------------------------------------------------------------------------------------- budget x
def _unit(*parts) -> float:
    """Deterministic uniform number in [0, 1) from the given parts."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def budget_mp(group_id: str, lo: float = 0.5, hi: float = MAX_PIXELS / 1e6, seed: int = 0) -> float:
    """Pixel budget x (MP) of a contrast set: log-uniform on [lo, hi]. Deterministic in (seed, group_id), so every
    version of a set gets the same x without any shared state."""
    if lo <= 0 or hi < lo:
        raise ValueError("need 0 < lo <= hi")
    u = _unit(seed, group_id, "x")
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


# ---------------------------------------------------------------------------------------------------------------- sizes
def resized_size(width: int, height: int, max_pixels: int, min_pixels: int = MIN_PIXELS) -> tuple:
    """(w, h) of an image under the pixel budget max_pixels, as smart_resize of qwen_vl_utils: sides rounded to
    multiples of PATCH, scaled down to fit max_pixels, scaled up to min_pixels."""
    h_bar = max(PATCH, round(height / PATCH) * PATCH)
    w_bar = max(PATCH, round(width / PATCH) * PATCH)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(PATCH, math.floor(height / beta / PATCH) * PATCH)
        w_bar = max(PATCH, math.floor(width / beta / PATCH) * PATCH)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / PATCH) * PATCH
        w_bar = math.ceil(width * beta / PATCH) * PATCH
    return int(w_bar), int(h_bar)


def magnified_size(width: int, height: int, max_pixels: int, magnify: float = MAGNIFY) -> tuple:
    """(w, h) a zoom crop is shown at: enlarged to fill the budget, by at most `magnify` per side, then on the patch
    grid within max_pixels. A crop larger than the budget is shrunk to it."""
    m = min(float(magnify), math.sqrt(max_pixels / (width * height))) if width > 0 and height > 0 else 1.0
    if m <= 1:
        return resized_size(width, height, max_pixels)
    return resized_size(max(1, round(width * m)), max(1, round(height * m)), max_pixels)


# ---------------------------------------------------------------------------------------------------------------- boxes
def view_to_orig(box: Sequence[float], view_size: Sequence[int], orig_size: Sequence[int]) -> list:
    """Box in pixels of the (downscaled) image the model saw -> pixels of the region it shows, clipped."""
    vw, vh = view_size
    ow, oh = orig_size
    sx, sy = ow / max(1, vw), oh / max(1, vh)
    x0, y0, x1, y1 = box
    return [min(max(0.0, x0 * sx), ow), min(max(0.0, y0 * sy), oh), min(max(0.0, x1 * sx), ow), min(max(0.0, y1 * sy), oh)]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def coverage(crop_box: Sequence[float], object_box: Sequence[float]) -> float:
    """Share of the object's box inside the crop (both in pixels of the photo)."""
    area = _area(object_box)
    if area <= 0:
        return 0.0
    inter = [max(crop_box[0], object_box[0]), max(crop_box[1], object_box[1]),
             min(crop_box[2], object_box[2]), min(crop_box[3], object_box[3])]
    return _area(inter) / area


# ------------------------------------------------------------------------------------------------- CrossRPO advantage
def crossrpo_advantages(groups: Sequence, versions: Sequence, correct: Sequence[float],
                        lam: float = CROSSRPO_LAMBDA) -> list:
    """CrossRPO advantage of each rollout, before the batch scale.

    Rollouts with the same group key belong to one contrast set (all its versions, one pixel budget); `versions`
    names the version of each rollout (the original O or an edited variant A / B); `correct` is its reward, the
    correctness c of its answer (1 or 0). For rollout i of version v with n rollouts:
        A_i = (1 + lam * w_v) * (c_i - mean of c over the other n-1 rollouts of version v),   w_v = prod_{u != v} p_u,
    where p_u is the share of correct rollouts of version u. The cross-input weight w_v estimates the probability
    that all other versions are answered correctly. The baseline uses only rollouts of the same version, and w_v only
    rollouts of the other versions. A single image (a set with one version) has w = 0, so its advantage is the
    leave-one-out advantage alone. A version with one rollout gets 0."""
    by: dict = {}
    for i, (g, v) in enumerate(zip(groups, versions)):
        by.setdefault(g, {}).setdefault(v, []).append(i)
    out = [0.0] * len(correct)
    for vs in by.values():
        p = {v: sum(float(correct[i]) for i in idx) / len(idx) for v, idx in vs.items()}
        for v, idx in vs.items():
            n = len(idx)
            if n < 2:
                continue
            w = math.prod(p[u] for u in vs if u != v) if len(vs) > 1 else 0.0
            total = sum(float(correct[i]) for i in idx)
            for i in idx:
                d = float(correct[i]) - (total - float(correct[i])) / (n - 1)
                out[i] = d + lam * w * d
    return out


# -------------------------------------------------------------------------------------------------------------- answers
_LETTER = re.compile(r"^\s*[\(\[\*]*\s*([A-H])\s*[\)\]\*]*(?:[\.\):\s]|$)")


def has_answer(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text.split("<answer>")[-1]


def parse_answer(text: str) -> str:
    """Option letter at the start of the last <answer>...</answer> ("" if none); markdown bold or brackets around the
    letter are allowed."""
    if not has_answer(text):
        return ""
    ans = text.split("<answer>")[-1].split("</answer>")[0]
    m = _LETTER.match(ans)
    return m.group(1) if m else ""
