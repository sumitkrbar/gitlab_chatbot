"""Retrieval + guardrail smoke test.

Measures two things that matter for a RAG bot:
* Retrieval hit-rate@k — for on-topic questions, does an expected source page
  appear in the top-k results?
* Guardrail precision — for off-topic questions, does the scope guardrail
  correctly refuse (weak retrieval)?

Run from the repo root:  python eval/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, guardrails  
from src.vectorstore import query  


def main() -> int:
    try:
        config.get_api_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    data = yaml.safe_load((Path(__file__).parent / "questions.yaml").read_text())
    questions = data["questions"]

    on_topic = [q for q in questions if not q.get("off_topic")]
    off_topic = [q for q in questions if q.get("off_topic")]

    print("=" * 70)
    print("RETRIEVAL HIT-RATE  (on-topic questions)")
    print("=" * 70)
    hits_count = 0
    for item in on_topic:
        results = query(item["q"])
        urls = [r["url"] for r in results]
        best = max((r["similarity"] for r in results), default=0.0)
        hit = any(item["expect_url_contains"] in u for u in urls)
        hits_count += hit
        mark = "[ OK ]" if hit else "[MISS]"
        print(f"{mark}  sim={best:.2f}  {item['q']}")
        if not hit:
            print(f"     expected '{item['expect_url_contains']}', got: {urls[:3]}")

    print("\n" + "=" * 70)
    print("GUARDRAIL  (off-topic should be refused)")
    print("=" * 70)
    refused_count = 0
    for item in off_topic:
        results = query(item["q"])
        verdict = guardrails.assess_retrieval(results)
        refused = not verdict.grounded
        refused_count += refused
        mark = "[ OK ]" if refused else "[MISS]"
        print(f"{mark}  sim={verdict.best_similarity:.2f}  {item['q']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Retrieval hit-rate@{config.TOP_K}: "
        f"{hits_count}/{len(on_topic)} = {hits_count / len(on_topic):.0%}"
    )
    print(
        f"Guardrail refusal on off-topic: "
        f"{refused_count}/{len(off_topic)} = {refused_count / len(off_topic):.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
