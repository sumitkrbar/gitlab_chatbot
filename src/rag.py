from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from google import genai
from google.genai import types

from . import config, guardrails
from .prompts import SYSTEM_PROMPT, build_user_turn

_client: genai.Client | None = None


def _genai() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.get_api_key())
    return _client


def _gen_config(**kwargs) -> types.GenerateContentConfig:
    
    if "2.5" in config.CHAT_MODEL:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


@dataclass
class RagResult:
    question: str 
    original_question: str
    hits: list[dict] = field(default_factory=list)
    grounded: bool = True
    refusal: str = ""
    best_similarity: float = 0.0
    rewritten: bool = False
    injection_flagged: bool = False


_PRONOUN_RE = re.compile(
    r"\b(it|its|that|this|those|these|they|them|their|he|she|the same)\b",
    re.IGNORECASE,
)


def _needs_rewrite(question: str, history: list[dict]) -> bool:
    
    if not history:
        return False
    words = question.split()
    return len(words) <= 6 or bool(_PRONOUN_RE.search(question))


def rewrite_followup(question: str, history: list[dict]) -> str:
    """Turn a context-dependent follow-up into a standalone search query."""
    recent = history[-config.HISTORY_TURNS :]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    prompt = (
        "Rewrite the user's follow-up into a single standalone question that "
        "makes sense without the chat history. Keep it faithful; do not answer "
        "it. Return only the rewritten question.\n\n"
        f"Chat history:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
    )
    try:
        resp = _genai().models.generate_content(
            model=config.CHAT_MODEL,
            contents=prompt,
            config=_gen_config(temperature=0.0, max_output_tokens=80),
        )
        rewritten = (resp.text or "").strip().strip('"')
        return rewritten or question
    except Exception: 
        return question


def retrieve(question: str, history: list[dict] | None = None) -> RagResult:
    """Run follow-up rewrite + retrieval + the scope/grounding guardrail."""
    from .vectorstore import query  

    history = history or []
    search_q = question
    rewritten = False
    if _needs_rewrite(question, history):
        search_q = rewrite_followup(question, history)
        rewritten = search_q.strip().lower() != question.strip().lower()

    hits = query(search_q)
    verdict = guardrails.assess_retrieval(hits)
    return RagResult(
        question=search_q,
        original_question=question,
        hits=hits,
        grounded=verdict.grounded,
        refusal=verdict.refusal,
        best_similarity=verdict.best_similarity,
        rewritten=rewritten,
    )


def _history_contents(history: list[dict]) -> list[types.Content]:
    out: list[types.Content] = []
    for m in history[-config.HISTORY_TURNS :]:
        role = "user" if m["role"] == "user" else "model"
        out.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )
    return out


def stream_answer(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> Iterator[str]:
    """Stream a grounded, cited answer. Yields incremental text chunks."""
    history = history or []
    contents = _history_contents(history)
    contents.append(
        types.Content(
            role="user", parts=[types.Part.from_text(text=build_user_turn(question, hits))]
        )
    )
    cfg = _gen_config(
        system_instruction=SYSTEM_PROMPT,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
    )
    try:
        for chunk in _genai().models.generate_content_stream(
            model=config.CHAT_MODEL, contents=contents, config=cfg
        ):
            if chunk.text:
                yield chunk.text
    except Exception as exc:  
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            reason = "the Gemini free-tier quota is exhausted for now"
        elif "503" in msg or "500" in msg or "ServerError" in type(exc).__name__:
            reason = "the model is briefly overloaded (a transient server error)"
        else:
            reason = f"an unexpected error occurred ({type(exc).__name__})"
        yield f"\n\n I couldn't finish that answer — {reason}. Please try again in a moment."
