"""Retrieve counselling-style replies from the local dataset (JSON lines: Context, Response)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings

_STOP = frozenset(
    """
    a an and are as at be been but by did do does for from had has have he her him his how
    i if in into is it its just me more my no not of on or our she so some than that the
    their them then there these they this to too up us was we were what when where which who
    why will with you your
    """.split()
)


def _dataset_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "data"
        / "mental health counselling conversations"
        / "combined_dataset.json"
    )


def _tokenize(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


@lru_cache(maxsize=1)
def _load_rows() -> list[dict[str, Any]]:
    path = _dataset_path()
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ctx = (obj.get("Context") or "").strip()
            resp = (obj.get("Response") or "").strip()
            if ctx and resp:
                rows.append({"Context": ctx, "Response": resp})
    return rows


def reply_from_dataset(user_message: str) -> str | None:
    """Pick the best-matching Response row by word overlap with Context; random if no overlap."""
    import random

    rows = _load_rows()
    if not rows:
        return None

    q_words = _tokenize(user_message)
    if not q_words:
        return random.choice(rows)["Response"]

    best_score = 0
    best_responses: list[str] = []
    for row in rows:
        ctx_words = _tokenize(row["Context"])
        score = len(q_words & ctx_words)
        if score > best_score:
            best_score = score
            best_responses = [row["Response"]]
        elif score == best_score and score > 0:
            best_responses.append(row["Response"])

    if best_score == 0:
        return random.choice(rows)["Response"]
    return random.choice(best_responses)
