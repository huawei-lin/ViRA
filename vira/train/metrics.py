"""Per-step training metrics (numpy only) and the rows that the advantage takes.

rollout_rows: one (set key, version) per rollout. The TransferQueue key {uid}_{session_id}_{index} gives the set
(uid), and the rollout's reward_extra_info gives its version; common.crossrpo_advantages groups the rollouts by them.
"""

from __future__ import annotations

import numpy as np

from vira import common as C

X_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]     # pixel budgets, in MP


def parse_key(key: str) -> tuple[str, int, int]:
    """TransferQueue key {uid}_{session_id}_{index} (uid is a uuid4 without underscores)."""
    uid, sid, idx = key.rsplit("_", 2)
    return uid, int(sid), int(idx)


def _info(e) -> dict:
    if isinstance(e, dict):
        return e.get("reward_extra_info", {}) or {}
    return {}


def _mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def rollout_rows(keys: list, extras: list) -> list:
    return [(parse_key(k)[0], int(_info(e).get("version", 0))) for k, e in zip(keys, extras, strict=True)]


def step_metrics(extras: list, prefix: str = "train") -> dict:
    """Rates over the given rollouts, each with its sample count (<prefix>/n_*), so a rate from a handful of rollouts
    is visible as such. The trainer also calls this on the last 10 steps pooled (prefix "train10")."""
    infos = [_info(e) for e in extras]
    if not infos or "acc" not in infos[0]:
        return {}
    P, X = prefix, prefix + "_x"
    a = {k: np.array([f.get(k, np.nan) for f in infos], dtype=np.float64) for k in infos[0]}
    src = np.array([(e.get("vira") or {}).get("source") or "?" if isinstance(e, dict) else "?" for e in extras])
    zoomed = a["n_ok_calls"] > 0
    m = {f"{P}/n": float(len(infos)), f"{P}/acc": _mean(a["acc"]), f"{P}/answered": _mean(a["answered"]),
         f"{P}/truncated": _mean(a["truncated"]), f"{P}/zoom_rate": _mean(zoomed), f"{P}/calls_mean": _mean(a["n_calls"]),
         f"{P}/tool_errors": _mean(a["n_errors"]), f"{P}/x_mean": _mean(a["x"])}

    def rate(name, values, sel):
        m[f"{P}/n_{name}"] = float(sel.sum())
        if sel.any():
            m[f"{P}/{name}"] = _mean(values[sel])

    boxed = a["has_box"] > 0.5                                # rows with a box of the object
    rate("hit_last", (a["cov_last"] >= C.HIT_COVERAGE).astype(float), zoomed & boxed)     # the last crop covers the object
    rate("acc_when_hit", a["acc"], boxed & (a["cov_max"] >= C.HIT_COVERAGE))              # some crop covers the object
    rate("multi_zoom", (a["n_ok_calls"] >= 2).astype(float), zoomed)
    for vi, name in enumerate("OAB"):
        rate(f"acc_{name}", a["acc"], (a["version"] == vi) & (a["n_versions"] == 3))
    variant_wrong = (a["n_versions"] >= 2) & (a["acc"] < 0.5) & (a["answered"] > 0.5) \
        & ~((a["n_versions"] == 3) & (a["version"] == 0))
    rate("chose_o_when_wrong", a["chose_o"], variant_wrong)   # wrong answers on a variant that are the original's
    for s in sorted(set(src)):
        rate(f"acc_src_{s}", a["acc"], src == s)
        rate(f"zoom_rate_src_{s}", zoomed.astype(float), src == s)
        rate(f"answered_src_{s}", a["answered"], src == s)
    rate("judge_failed", a["judge_failed"], (a["open"] > 0.5) & (a["answered"] > 0.5))    # open questions
    for lo, hi in X_BUCKETS:
        sel = (a["x"] >= lo) & (a["x"] < hi)
        tag = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        m[f"{X}/n_{tag}"] = float(sel.sum())
        if sel.any():
            m[f"{X}/zoom_rate_{tag}"] = _mean(zoomed[sel])
            m[f"{X}/acc_{tag}"] = _mean(a["acc"][sel])
            m[f"{X}/calls_{tag}"] = _mean(a["n_calls"][sel])
    return {k: v for k, v in m.items() if v == v}  # drop NaN
