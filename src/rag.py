"""Ask questions against the local Chroma RAG index."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

try:
    from ingest import (
        DEFAULT_CHROMA_PATH,
        DEFAULT_COLLECTION_NAME,
        DEFAULT_EMBEDDING_MODEL,
        embed_texts,
    )
except ImportError:  # pragma: no cover - supports package-style imports
    from .ingest import (
        DEFAULT_CHROMA_PATH,
        DEFAULT_COLLECTION_NAME,
        DEFAULT_EMBEDDING_MODEL,
        embed_texts,
    )


load_dotenv()

DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_MIN_SCORE = 0.25

MAX_QUERY_LENGTH = 500       # characters — questions longer than this are rejected
MAX_ANSWER_TOKENS = 512      # tokens — caps how long the generated answer can be
MAX_QUERIES_PER_MINUTE = 10  # rate limit — max questions per 60-second window
MAX_SESSION_QUERIES = 20     # soft cost cap — interactive mode stops after this many questions


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its similarity score."""

    id: str
    source: str
    text: str
    score: float
    metadata: dict


def require_openai_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to a local .env file or set it in your shell. "
            "Do not commit the real key to GitHub."
        )


class RAGPipeline:
    """Semantic retrieval plus short answer generation."""

    def __init__(
        self,
        chroma_path: Path | str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.collection = self.load_collection()

    def load_collection(self):
        if not self.chroma_path.exists():
            raise FileNotFoundError(
                f"Chroma database not found: {self.chroma_path}. Run `python src/ingest.py` first."
            )

        client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            return client.get_collection(self.collection_name)
        except ValueError as exc:
            raise FileNotFoundError(
                f"Chroma collection not found: {self.collection_name}. Run `python src/ingest.py` first."
            ) from exc

    def retrieve(self, query: str, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> list[SearchResult]:
        """Return chunks most semantically similar to the query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = embed_texts([query], model=self.embedding_model)[0]
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for result_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance)
            if score < min_score:
                continue

            results.append(
                SearchResult(
                    id=result_id,
                    source=str(metadata.get("source", "unknown")),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def answer(self, query: str, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> str:
        """Generate a concise answer from retrieved context."""

        if len(query) > MAX_QUERY_LENGTH:
            return (
                f"Your question is too long ({len(query)} characters). "
                f"Please keep it under {MAX_QUERY_LENGTH} characters."
            )

        results = self.retrieve(query, top_k, min_score)
        if not results:
            return "I could not find relevant indexed context for that question."

        answer = self.generate_answer(query, results)
        return f"{answer}\n\n{format_sources(results)}"

    def generate_answer(self, query: str, results: list[SearchResult]) -> str:
        """Ask the chat model for a focused answer grounded in retrieved chunks."""

        require_openai_key()
        client = OpenAI()
        context = format_context(results)
        prompt = f"""Answer the user's question using only the retrieved context.

Keep the answer short and focused. If the context is insufficient, say that the indexed papers do not provide enough information.

Question:
{query}

Retrieved context:
{context}
"""
        response = client.responses.create(
            model=self.chat_model,
            input=prompt,
            max_output_tokens=MAX_ANSWER_TOKENS,
        )
        return response.output_text.strip()


def format_context(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.metadata.get("title", "")
        page = result.metadata.get("page", "")
        lines.append(f"[{index}] Source: {result.source}; Title: {title}; Page: {page}; Score: {result.score:.3f}")
        lines.append(result.text)
        lines.append("")
    return "\n".join(lines).strip()


def format_sources(results: list[SearchResult]) -> str:
    lines = ["Sources:"]
    seen: set[tuple[str, int]] = set()
    for result in results:
        page = int(result.metadata.get("page", 0))
        key = (result.source, page)
        if key in seen:
            continue
        seen.add(key)
        page_text = f", page {page}" if page else ""
        lines.append(f"- {result.source}{page_text} (score: {result.score:.3f})")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against a semantic local RAG index.")
    parser.add_argument("query", nargs="?", help="Question to ask. Omit to enter interactive mode.")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Local Chroma database folder.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks without generating an answer.")
    return parser.parse_args()


def run_interactive(pipeline: RAGPipeline, top_k: int, min_score: float, show_context: bool) -> None:
    print("RAG ready. Type your question and press Enter. Type 'quit' to exit.")
    print(f"Limits: {MAX_QUERY_LENGTH} chars per question, {MAX_QUERIES_PER_MINUTE} questions/minute, {MAX_SESSION_QUERIES} questions per session.\n")

    session_count = 0
    recent_timestamps: deque[float] = deque()

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        # Session cap
        if session_count >= MAX_SESSION_QUERIES:
            print(f"Session limit reached ({MAX_SESSION_QUERIES} questions). Restart the script to continue.")
            break

        # Rate limit: drop timestamps older than 60 seconds, then check count
        now = time.monotonic()
        while recent_timestamps and now - recent_timestamps[0] > 60:
            recent_timestamps.popleft()
        if len(recent_timestamps) >= MAX_QUERIES_PER_MINUTE:
            wait = 60 - (now - recent_timestamps[0])
            print(f"Too many questions. Please wait {wait:.0f} seconds.")
            continue

        recent_timestamps.append(now)
        session_count += 1

        results = pipeline.retrieve(query, top_k, min_score)
        if show_context:
            print(format_context(results))
        else:
            print(pipeline.answer(query, top_k, min_score))
        print(f"[{session_count}/{MAX_SESSION_QUERIES} questions used this session]\n")


def main() -> None:
    args = parse_args()
    pipeline = RAGPipeline(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )

    if args.query:
        results = pipeline.retrieve(args.query, args.top_k, args.min_score)
        if args.show_context:
            print(format_context(results))
        else:
            print(pipeline.answer(args.query, args.top_k, args.min_score))
    else:
        run_interactive(pipeline, args.top_k, args.min_score, args.show_context)


if __name__ == "__main__":
    main()
