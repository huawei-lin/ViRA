"""Answer parsing for evaluation.

Two readings of a final reply:

- strict: the option letter inside ``<answer>...</answer>`` (``common.parse_answer``), the reading the training
  reward uses;
- lenient: :func:`parse_letter`, the option letter read from free text as well (``Answer: B``, a lone ``(B)`` on the
  last line, the text of exactly one option, ...), the reading every model in the offline evaluation is scored with.
"""

from __future__ import annotations

import re

from vira import common as C

LET = "ABCDEFGH"


def answer_text(t: str | None) -> str | None:
    """Content of the last ``<answer>...</answer>``, or None."""
    m = re.findall(r"<answer>(.*?)</answer>", t or "", flags=re.S)
    return m[-1].strip() if m else None


def strip_think(t: str | None) -> str:
    """Drop ``<think>...</think>`` blocks (and anything before a lone ``</think>``)."""
    t = re.sub(r"<think>.*?</think>", " ", t or "", flags=re.S)
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return t


def norm(s: str | None) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def parse_letter(text: str | None, options) -> str:
    """Lenient reading: the letter of the chosen option, or '' when none can be read."""
    a = answer_text(text)
    t = (a if a is not None else strip_think(text)).strip()
    plain = re.sub(r"[*_`]", "", text or "")
    tp = re.sub(r"[*_`]", "", t).strip()
    n = len(options)
    ok = lambda x: LET.index(x.upper()) < n  # noqa: E731
    if a is None:
        m = re.findall(r"(?:final answer|correct answer|answer|correct option|correct choice)\s*(?:is|:|\uff1a|=)\s*\(?\s*([A-H])\b",
                       plain, flags=re.I)
        m = [x.upper() for x in m if ok(x)]
        if m:
            return m[-1]
        lines = [l.strip() for l in tp.splitlines() if l.strip()]  # noqa: E741
        if lines:
            m = re.match(r"^\(?\s*([A-H])\s*\)?\s*[.:]?$", lines[-1])
            if m and ok(m.group(1)):
                return m.group(1)
    m = re.match(r"^\(?\s*([A-H])\s*(?:[\).:\]]|$)", tp)
    if m and ok(m.group(1)) and (len(tp) <= 40 or a is not None):
        return m.group(1)
    hits = [i for i, o in enumerate(options) if norm(o) and norm(o) == norm(tp)]
    if len(hits) == 1:
        return LET[hits[0]]
    hits = [i for i, o in enumerate(options) if norm(o) and len(norm(o)) >= 2 and norm(o) in norm(tp)]
    if len(hits) == 1:
        return LET[hits[0]]
    m = re.findall(r"\b([A-H])\b", tp)
    m = [x for x in m if ok(x)]
    return m[-1] if len(set(m)) == 1 else ""


def parse_strict(text: str | None) -> str:
    """Strict reading: the letter inside <answer></answer> ('' if none), as in the training reward."""
    return C.parse_answer(text or "") or ""
