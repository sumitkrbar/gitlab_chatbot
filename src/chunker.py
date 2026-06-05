"""Heading-aware, token-bounded chunking.

We first split a page on Markdown-ish headings (trafilatura emits ``###`` style
headers), then pack consecutive sections into ~CHUNK_TOKENS windows with a small
overlap. Each chunk keeps its source URL, page title, and the nearest heading so
the UI can show precise, clickable citations.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import tiktoken

from . import config
from .scraper import Page

_enc = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    url: str
    title: str
    heading: str
    metadata: dict = field(default_factory=dict)


def _ntokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf:
                sections.append((current_heading, "\n".join(buf).strip()))
                buf = []
            current_heading = m.group(2).strip()
        else:
            buf.append(line)
    if buf:
        sections.append((current_heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def _pack(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    
    target = config.CHUNK_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS
    packed: list[tuple[str, str]] = []

    buf_heading = ""
    buf_parts: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_heading, buf_parts, buf_tokens
        if buf_parts:
            packed.append((buf_heading, "\n\n".join(buf_parts).strip()))
        buf_heading, buf_parts, buf_tokens = "", [], 0

    for heading, body in sections:
        toks = _enc.encode(body)
        if len(toks) > target: 
            flush()
            start = 0
            while start < len(toks):
                window = toks[start : start + target]
                packed.append((heading, _enc.decode(window)))
                if start + target >= len(toks):
                    break
                start += target - overlap
            continue

        if buf_tokens + len(toks) > target:
            flush()
        if not buf_parts:
            buf_heading = heading
        buf_parts.append(f"## {heading}\n{body}" if heading else body)
        buf_tokens += len(toks)

    flush()
    return packed


def chunk_page(page: Page) -> list[Chunk]:
    sections = _split_into_sections(page.text) or [("", page.text)]
    chunks: list[Chunk] = []
    for heading, body in _pack(sections):
        breadcrumb = f"{page.title} > {heading}".strip(" >")
        embed_text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
        cid = hashlib.sha1(
            f"{page.url}|{heading}|{body[:120]}".encode()
        ).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=cid,
                text=embed_text,
                url=page.url,
                title=page.title,
                heading=heading,
                metadata={
                    "url": page.url,
                    "title": page.title,
                    "heading": heading,
                    "tokens": _ntokens(body),
                },
            )
        )
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(page))
    return out
