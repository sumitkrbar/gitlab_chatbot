"""Curated breadth-first crawler for GitLab's Handbook + Direction pages.

Design choices:
- We crawl only URLs under ``ALLOWED_PREFIXES`` (the curated subset for now). This keeps
  answer quality high and the index small enough to commit to the repo.
- Main-content extraction uses ``trafilatura`` (boilerplate removal) with a
  BeautifulSoup fallback, so chunks are clean prose rather than nav/footer noise.
- Everything is bounded (``MAX_PAGES``, ``MAX_DEPTH``) and we **log skips** rather
  than silently truncating, so the ingest summary is honest about coverage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from . import config


@dataclass
class Page:
    url: str
    title: str
    text: str


def _is_allowed(url: str) -> bool:
    return any(url.startswith(p) for p in config.ALLOWED_PREFIXES)


def _normalise(url: str) -> str:
    """Drop fragments and trailing slashes so we don't crawl dupes."""
    url, _frag = urldefrag(url)
    if url.endswith("/") and len(urlparse(url).path) > 1:
        url = url.rstrip("/")
    return url


def _extract(html: str, url: str) -> tuple[str, str]:
    """Return (title, clean_text). Empty text means 'nothing useful here'."""
    # markdown output preserves '#' headings so the chunker can split on sections.
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        output_format="markdown",
    )
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    title = title.replace(" | GitLab", "").strip()
    if not text:  # fallback: strip scripts/styles and take visible text
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        main = soup.find("main") or soup.body or soup
        text = main.get_text("\n", strip=True)
    return title, (text or "").strip()


def _links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        absolute = _normalise(urljoin(base_url, a["href"]))
        if absolute.startswith("http") and _is_allowed(absolute):
            out.append(absolute)
    return out


def _fetch(client: httpx.Client, url: str) -> str | None:
    """Return HTML for an allowed, successful HTML response, else None."""
    try:
        resp = client.get(url)
        if resp.status_code != 200 or "html" not in resp.headers.get(
            "content-type", ""
        ):
            return None
        return resp.text
    except Exception:  # noqa: BLE001 - network is best-effort
        return None


def _crawl_seed(
    client: httpx.Client,
    seed: str,
    confine: str,
    page_cap: int,
    max_depth: int,
    seen: set[str],
) -> tuple[list[Page], int, int]:
    pages: list[Page] = []
    skipped = errors = 0
    frontier = [seed]
    for depth in range(max_depth + 1):
        if not frontier or len(pages) >= page_cap:
            break
        frontier = frontier[: (page_cap - len(pages)) * 2]  # don't over-fetch
        with ThreadPoolExecutor(max_workers=config.CRAWL_WORKERS) as pool:
            htmls = list(pool.map(lambda u: _fetch(client, u), frontier))

        next_frontier: list[str] = []
        for url, html in zip(frontier, htmls):
            if html is None:
                errors += 1
                continue
            title, text = _extract(html, url)
            if len(text) < 200: 
                skipped += 1
            else:
                pages.append(Page(url=url, title=title, text=text))
                if len(pages) >= page_cap:
                    break
            if depth < max_depth:
            
                for link in _links(html, url):
                    if link.startswith(confine) and link not in seen:
                        seen.add(link)
                        next_frontier.append(link)
        frontier = next_frontier
    return pages, skipped, errors


def crawl(
    seeds: list[str] | None = None,
    max_pages: int | None = None,
    max_depth: int | None = None,
    verbose: bool = True,
) -> list[Page]:
    """Crawl each seed's subtree with a fair per-section page budget.

    Crawling all seeds from one shared frontier let a link-heavy section (e.g.
    /handbook/company) starve the rest. Instead we give each seed an equal slice
    of the page budget and confine its BFS to its own URL subtree, so product,
    engineering, values, etc. all get represented. Within a seed, each BFS level
    is fetched concurrently (handbook.gitlab.com is ~3s/page, so a thread pool
    cuts the build from minutes to well under one).
    """
    seeds = [_normalise(u) for u in (seeds or config.SEED_URLS)]
    max_pages = max_pages or config.MAX_PAGES
    max_depth = max_depth if max_depth is not None else config.MAX_DEPTH
    per_seed = max(1, -(-max_pages // len(seeds))) 

    seen: set[str] = set()
    pages: list[Page] = []
    skipped_short = 0
    errors = 0

    headers = {"User-Agent": config.USER_AGENT}
    with httpx.Client(
        headers=headers, timeout=config.REQUEST_TIMEOUT, follow_redirects=True
    ) as client:
        for seed in seeds:
            if seed in seen:
                continue
            seen.add(seed)
            cap = min(per_seed, max_pages - len(pages))
            if cap <= 0:
                break
            sec_pages, sk, er = _crawl_seed(
                client, seed, confine=seed, page_cap=cap, max_depth=max_depth, seen=seen
            )
            pages.extend(sec_pages)
            skipped_short += sk
            errors += er
            if verbose:
                print(f"  {seed}  ->  {len(sec_pages)} pages")

    if verbose:
        print(
            f"Crawl done: {len(pages)} pages kept, "
            f"{skipped_short} skipped (too short), {errors} errors."
        )
    return pages
