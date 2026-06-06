"""Input hygiene + retrieval-based scope/grounding guardrails.

Kept simple and deterministic on purpose. A classifier could do better on edge
cases, but rules are easy to reason about and don't add a model dependency.

- check_input: reject empty / over-long / obvious prompt-injection inputs before
  spending an API call.
- assess_retrieval: if the best similarity score is weak, the question is probably
  off-topic or not covered, so refuse instead of letting the model hallucinate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the above)",
    r"disregard (all|the|your) (instructions|rules)",
    r"system prompt",
    r"you are now",
    r"act as (?:an?|the)",
    r"reveal your (prompt|instructions)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class InputVerdict:
    ok: bool
    message: str = ""  # user-facing reason when ok is False
    injection_flagged: bool = False


def check_input(text: str) -> InputVerdict:
    stripped = (text or "").strip()
    if not stripped:
        return InputVerdict(ok=False, message="Please type a question.")
    if len(stripped) > MAX_INPUT_CHARS:
        return InputVerdict(
            ok=False,
            message=(
                f"That message is a bit long ({len(stripped)} chars). "
                f"Please keep questions under {MAX_INPUT_CHARS} characters."
            ),
        )
    return InputVerdict(ok=True, injection_flagged=bool(_INJECTION_RE.search(stripped)))


@dataclass
class RetrievalVerdict:
    grounded: bool
    best_similarity: float
    refusal: str = ""  # populated when grounded is False


OFF_TOPIC_REFUSAL = (
    "I'm the **GitLab Handbook Assistant**, so I can only answer questions grounded "
    "in GitLab's public Handbook and Direction pages — things like GitLab's values, "
    "culture, processes, engineering practices, and product direction.\n\n"
    "I couldn't find anything relevant to that question in those sources. "
    "Try asking about, for example:\n"
    + "\n".join(f"- {q}" for q in config.STARTER_QUESTIONS)
)


def assess_retrieval(hits: list[dict]) -> RetrievalVerdict:
    """Decide whether retrieval is strong enough to attempt a grounded answer."""
    best = max((h["similarity"] for h in hits), default=0.0)
    if not hits or best < config.MIN_SIMILARITY:
        return RetrievalVerdict(
            grounded=False, best_similarity=best, refusal=OFF_TOPIC_REFUSAL
        )
    return RetrievalVerdict(grounded=True, best_similarity=best)
