"""Evaluation metrics: one schema for the validation inside training and the offline evaluation, one
``compute_metrics``.

Episode = one (question, version, budget) answered once:

    qid, version ("O"/"A"/"B": the original and the two edited variants; a single image is "O"),
    budget ("0.5mp", "1mp", "2mp", "4mp", "maxmp"), strict (bool), lenient (bool), answered (an answer is present),
    zooms (successful zoom calls), hit (the last crop covers >= 0.5 of the object's box; None without zooms), cov_max,
    letter, gold, gold_o (correct letter of the original), tokens (prompt + completion tokens summed over the calls,
    offline evaluation only), seq_tokens (final sequence length, in training only), source (the type of the question
    in ViRA-Train, in training only), open (an open question, judged)

Sources: the validation inside training (:func:`episode_from_rollout`) and rows of the offline evaluation
``vira.eval.run`` (:func:`episode_from_row`). Metric names (per budget tag b):

    {p}/set_acc_strict_b, {p}/set_acc_lenient_b   set accuracy: all three versions correct (sets with three versions)
    {p}/acc_b, {p}/acc_{O,A,B}_b                   per-image accuracy (strict)
    {p}/hit_last_b                                 zoomed episodes whose last crop covers the object
    {p}/acc_when_hit_b                             strict accuracy when any crop covers the object (cov_max >= 0.5)
    {p}/chose_o_when_wrong_b                       wrong answers on A/B that are the original's answer
    {p}/zooms_b, {p}/answered_b, {p}/n_questions_b, {p}/tokens_b or {p}/seq_tokens_b
Rates are percentages.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from vira import common as C

from .parsing import parse_letter, parse_strict

Episode = dict[str, Any]


def budget_tag(x) -> str:
    """Budget in MP (or "max") -> metric tag: 1.0 -> "1mp", 0.5 -> "0.5mp", "max" or anything above 12 MP -> "maxmp"."""
    if x == "max" or float(x) > 12:
        return "maxmp"
    return f"{float(x):g}mp"


# --------------------------------------------------------------------------------------------------- normalizers
def episode_from_rollout(extra: Mapping[str, Any], seq_tokens: int | None = None,
                         prompt_tokens: int | None = None) -> Episode:
    """Validation inside training: the ``extra_fields`` of one agent-loop output (``vira`` + ``reward_extra_info``).
    Besides the metric fields it keeps the final reply, the zoom boxes (pixels of the photo) and the token count of
    the first prompt."""
    v, info = extra["vira"], extra["reward_extra_info"]
    reply = v.get("last_reply") if v.get("last_reply") is not None else v.get("final_text", "")
    gold = v["gold"]
    is_open = bool(info.get("open", 0.0))
    if is_open:      # open question: correct = the judgment of the <answer> text (the agent loop's acc)
        strict = lenient = float(info.get("acc", 0.0)) > 0.5
        answered, strict_letter, lenient_letter = bool(info.get("answered", 0.0)), v.get("answer", ""), ""
    else:
        strict_letter = parse_strict(reply)
        lenient_letter = parse_letter(reply, v.get("options") or [])
        strict, lenient, answered = strict_letter == gold, lenient_letter == gold, bool(strict_letter)
    zooms = int(info.get("n_ok_calls", 0))
    return {"qid": v["qid"], "version": str(v["version"]).upper(), "budget": budget_tag(v["x"]),
            "strict": strict, "lenient": lenient, "open": is_open, "source": v.get("source"),
            "answered": answered, "zooms": zooms,
            "hit": (info.get("cov_last", 0.0) >= C.HIT_COVERAGE) if zooms else None,
            "cov_max": float(info.get("cov_max", 0.0)), "letter": strict_letter or lenient_letter, "gold": gold,
            "gold_o": (v.get("golds") or {}).get("O"), "tokens": None, "seq_tokens": seq_tokens,
            "x": float(v["x"]), "truncated": bool(info.get("truncated", 0.0)), "prompt_tokens": prompt_tokens,
            "boxes": [c["bbox"] for c in v.get("calls") or [] if c.get("bbox")], "reply": reply}


def episode_from_row(row: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]] | None = None) -> Episode:
    """A row of the offline evaluation (``vira.eval.run``): correct = the lenient parse_letter; strict = the
    <answer> letter of the final reply. ``items`` (qid -> item) gives the object's box and the original's answer."""
    it = (items or {}).get(row["qid"], {})
    zs = [z for z in row.get("zooms") or [] if z.get("bbox")]
    box = it.get("diff_bbox")
    covs = [C.coverage(z["bbox"], box) for z in zs] if box else []
    strict_letter = parse_strict((row.get("replies") or [""])[-1])
    gold = row["gold"]
    return {"qid": row["qid"], "version": row["version"].upper(), "budget": budget_tag(row["budget"]),
            "strict": strict_letter == gold, "lenient": bool(row.get("correct")), "answered": bool(strict_letter),
            "zooms": len(zs), "hit": (covs[-1] >= C.HIT_COVERAGE) if covs else None,
            "cov_max": max(covs) if covs else 0.0, "letter": strict_letter or row.get("pred", ""), "gold": gold,
            "gold_o": (it.get("gold") or {}).get("o"),
            "tokens": row.get("prompt_tokens_sum", 0) + row.get("completion_tokens", 0), "seq_tokens": None}


# ------------------------------------------------------------------------------------------------------- metrics
def _pct(xs: Iterable[bool]) -> float:
    xs = list(xs)
    return 100.0 * sum(bool(x) for x in xs) / len(xs) if xs else 0.0


def budget_metrics(episodes: list[Episode], prefix: str) -> dict[str, float]:
    """Per-budget metrics (see the module docstring for the names)."""
    by: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by[e["budget"]].append(e)
    m: dict[str, float] = {}
    for b, rs in by.items():
        q: dict[str, list[Episode]] = defaultdict(list)
        for r in rs:
            q[r["qid"]].append(r)
        full = [v for v in q.values() if len(v) == 3]
        if full:
            m[f"{prefix}/set_acc_strict_{b}"] = _pct(all(x["strict"] for x in v) for v in full)
            m[f"{prefix}/set_acc_lenient_{b}"] = _pct(all(x["lenient"] for x in v) for v in full)
            m[f"{prefix}/n_questions_{b}"] = float(len(full))
        m[f"{prefix}/acc_{b}"] = _pct(r["strict"] for r in rs)
        for v in "OAB":
            sel = [r for r in rs if r["version"] == v]
            if sel:
                m[f"{prefix}/acc_{v}_{b}"] = _pct(r["strict"] for r in sel)
        zoomed = [r for r in rs if r["zooms"]]
        if zoomed:
            m[f"{prefix}/hit_last_{b}"] = _pct(r["hit"] for r in zoomed)
        hit = [r for r in rs if r["cov_max"] >= C.HIT_COVERAGE]
        if hit:
            m[f"{prefix}/acc_when_hit_{b}"] = _pct(r["strict"] for r in hit)
        wrong_variant = [r for r in rs if r["version"] != "O" and r["answered"] and not r["strict"] and r["letter"]]
        if wrong_variant:
            m[f"{prefix}/chose_o_when_wrong_{b}"] = _pct(r["letter"] == r["gold_o"] for r in wrong_variant)
        m[f"{prefix}/zooms_{b}"] = sum(r["zooms"] for r in rs) / len(rs)
        m[f"{prefix}/answered_{b}"] = _pct(r["answered"] for r in rs)
        if all(r.get("tokens") is not None for r in rs):
            m[f"{prefix}/tokens_{b}"] = sum(r["tokens"] for r in rs) / len(rs)
        if all(r.get("seq_tokens") is not None for r in rs):
            m[f"{prefix}/seq_tokens_{b}"] = sum(r["seq_tokens"] for r in rs) / len(rs)
    return m


def overall_acc(episodes: list[Episode], prefix: str) -> dict[str, float]:
    """Per-image accuracy over every validation image (versions of contrast sets + single images) per budget, and its
    mean over the budgets ({p}/acc_all_mean)."""
    by: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by[e["budget"]].append(e)
    m = {f"{prefix}/acc_all_{b}": _pct(r["strict"] for r in rs) for b, rs in by.items()}
    if m:
        m[f"{prefix}/acc_all_mean"] = sum(m.values()) / len(m)
    return m


# the single-image types of ViRA-Train: "detail" (open questions about one detail of a photo) and the other types
DETAIL = "detail"
OTHER_SINGLE = ("multi_evidence_photo", "multi_evidence_map", "chart", "math")


def dev_group(e: Episode) -> str:
    """"set" (contrast sets) or the type of a single-image question."""
    s = e.get("source")
    return s if s == DETAIL or s in OTHER_SINGLE else "set"


def macro_acc(groups: dict[str, list[Episode]], prefix: str) -> dict[str, float]:
    """Per budget, the mean of three accuracies: contrast sets, detail questions and the other single-image types
    pooled ("other"), each counted once whatever its number of episodes ({p}/acc_macro_{b}); {p}/acc_macro_mean = its
    mean over the budgets. Only written when the other single-image types are in the validation set."""
    parts = {"set": groups.get("set", []), DETAIL: groups.get(DETAIL, []),
             "other": [e for s in OTHER_SINGLE for e in groups.get(s, [])]}
    if not parts["other"]:
        return {}
    by: dict[str, list[float]] = defaultdict(list)
    for eps in parts.values():
        per_b: dict[str, list[Episode]] = defaultdict(list)
        for e in eps:
            per_b[e["budget"]].append(e)
        for b, rs in per_b.items():
            by[b].append(_pct(r["strict"] for r in rs))
    m = {f"{prefix}/acc_macro_{b}": sum(v) / len(v) for b, v in by.items()}
    m[f"{prefix}/acc_macro_mean"] = sum(m.values()) / len(m)
    m.update({f"{prefix}_other/acc_{b}": _pct(r["strict"] for r in parts["other"] if r["budget"] == b) for b in by
              if any(r["budget"] == b for r in parts["other"])})
    return m


def compute_metrics(episodes: Iterable[Episode], prefix: str = "dev") -> dict[str, float]:
    """All metrics of the episodes. {prefix}/ = contrast sets, {prefix}_<type>/ = each single-image type;
    {prefix}/acc_all_* = every episode pooled, {prefix}/acc_macro_* = contrast sets, detail questions and the other
    single-image types weighted equally."""
    episodes = list(episodes)
    groups: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        groups[dev_group(e)].append(e)
    out = budget_metrics(groups.get("set", []), prefix)
    for g, eps in groups.items():
        if g != "set":
            out.update(budget_metrics(eps, f"{prefix}_{g}"))
    out.update(overall_acc(episodes, prefix))
    out.update(macro_acc(groups, prefix))
    return out
