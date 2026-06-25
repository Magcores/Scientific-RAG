"""Evaluate RAG answer faithfulness using OpenAI as the judge.

Faithfulness measures whether each claim in the generated answer is actually
supported by the retrieved context chunks. This is the standard way to detect
hallucinations in a RAG system.

How it works (two LLM calls per question):
  1. Ask the model to break the generated answer into individual factual claims.
  2. For each claim, ask whether it is supported by the retrieved chunks.
  Score = supported claims / total claims

A score of 1.0 means every claim traces back to the sources.
Lower scores mean the model added content that was not in the documents.

Usage:
    python src/faithfulness.py

Requires the Chroma index to be built first:
    python src/ingest.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
from openai import OpenAI

from rag import RAGPipeline, SearchResult

load_dotenv()

# Questions to evaluate. These should reflect what users would actually ask
# against your indexed documents. Adjust to match your papers.
SAMPLE_QUESTIONS = [
    "What is the role of theta activity in cognitive functioning?",
    "How does ADHD affect EEG patterns?",
    "What is the relationship between theta waves and working memory?",
    "How do researchers measure theta activity in EEG studies?",
    "What are the differences in EEG between ADHD and non-ADHD individuals?",
    "How does age affect theta activity in children and adolescents?",
    "What methods are used to analyze EEG data in cognitive research?",
    "What brain regions are associated with theta oscillations?",
]


def extract_claims(client: OpenAI, answer: str, model: str) -> list[str]:
    """Ask the model to break an answer into individual factual claims."""

    prompt = f"""Break the following answer into a list of individual factual claims.
Each claim should be a single, self-contained statement.
Return a JSON array of strings. Return only the JSON, no explanation.

Answer:
{answer}"""

    response = client.responses.create(model=model, input=prompt)
    raw = response.output_text.strip()

    # Strip markdown code fences if the model wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        claims = json.loads(raw.strip())
        return [c for c in claims if isinstance(c, str) and c.strip()]
    except json.JSONDecodeError:
        # Fallback: split by newline if JSON parsing fails
        return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]


def check_claim(client: OpenAI, claim: str, context: str, model: str) -> bool:
    """Ask the model whether a single claim is supported by the context."""

    prompt = f"""Does the following context support this claim?
Answer with exactly one word: YES or NO.

Claim: {claim}

Context:
{context}"""

    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip().upper().startswith("YES")


def score_faithfulness(
    client: OpenAI,
    answer: str,
    results: list[SearchResult],
    model: str,
) -> tuple[float, list[str], list[bool]]:
    """Return (score, claims, supported_flags) for one answer."""

    claims = extract_claims(client, answer, model)
    if not claims:
        return 0.0, [], []

    context = "\n\n".join(r.text for r in results)
    supported = [check_claim(client, claim, context, model) for claim in claims]
    score = sum(supported) / len(supported)
    return score, claims, supported


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to a .env file or set it in your shell."
        )

    client = OpenAI()
    judge_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    print("Loading RAG pipeline...")
    pipeline = RAGPipeline()

    print(f"\nRunning {len(SAMPLE_QUESTIONS)} questions through the pipeline...\n")

    all_scores: list[float] = []

    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")

        results = pipeline.retrieve(question)
        if not results:
            print("   Skipped — no chunks retrieved above the score threshold.\n")
            continue

        answer = pipeline.generate_answer(question, results)
        print(f"A: {answer[:120]}{'...' if len(answer) > 120 else ''}")

        score, claims, supported = score_faithfulness(client, answer, results, judge_model)
        all_scores.append(score)

        print(f"\n   Faithfulness: {score:.2f}  ({sum(supported)}/{len(claims)} claims supported)")
        for claim, is_supported in zip(claims, supported):
            mark = "OK" if is_supported else "!!"
            print(f"   [{mark}] {claim}")
        print()

    if not all_scores:
        print("No questions were evaluated. Is the Chroma database built? Run `python src/ingest.py` first.")
        return

    overall = sum(all_scores) / len(all_scores)
    bar = "#" * int(overall * 40)
    print(f"{'─' * 60}")
    print(f"Overall faithfulness: {overall:.3f} / 1.000")
    print(f"[{bar:<40}]")
    print(
        "\n1.00 = every claim in every answer was supported by the retrieved chunks."
        "\nLower = the model added things that weren't in the sources."
    )


if __name__ == "__main__":
    main()
