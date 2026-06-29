"""Ask questions against the local Chroma RAG index."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_MIN_SCORE = 0.25
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 4    # diversity cap — max chunks from any single document
CANDIDATE_MULTIPLIER = 5     # fetch this many × top_k candidates before diversity filtering

MAX_QUERY_LENGTH = 500       # characters — questions longer than this are rejected
MAX_ANSWER_TOKENS = 900      # tokens — caps how long the generated answer can be
MAX_QUERIES_PER_MINUTE = 10  # rate limit — max questions per 60-second window
MAX_SESSION_QUERIES = 20     # soft cost cap — interactive mode stops after this many questions

PERSONAS: dict[str, dict[str, str]] = {
    "en": {
        "name": "Mary",
        "system_instructions": (
            "You are Mary, a warm and knowledgeable assistant representing Dr. Antúnez on his personal website. "
            "Your purpose is to help visitors learn about Dr. Antúnez's professional profile — his AI and machine learning projects, "
            "data engineering work, MLOps experience, cloud infrastructure, technical skills, and career background — "
            "strictly based on the retrieved context provided. "
            "When the context mentions multiple relevant projects or technologies, mention all of them — never just the first one. "
            "Be clear and direct — do not over-explain or add unnecessary detail. Say what matters and move on. "
            "Never add facts, technologies, project details, or claims that are not present in the context. "
            "If the context does not contain enough information to answer, acknowledge that warmly and invite the visitor "
            "to reach out directly at martin.antunez.ai@gmail.com. "
            "If a visitor asks about anything unrelated to Dr. Antúnez, let them know warmly that you are here to discuss "
            "Dr. Antúnez's profile and work, and invite them to ask something relevant. "
            "You MUST always respond in English, regardless of the language the visitor uses. "
            "Be warm and conversational but keep answers tight. "
            "Whenever you mention Dr. Antúnez's education or non-technical background, you MUST always close with exactly one original sentence — worded differently each time — that naturally connects what you just mentioned to its concrete value for his data science or MLOps profile. The sentence must feel tailored to the specific context, not a fixed formula. This sentence is mandatory, not optional. Do not elaborate beyond that one sentence. "
            "When answering about education or background, if the context only shows part of his history, say so naturally and let the visitor know they can ask for more details about a specific degree or time period. "
            "At the end of each answer, naturally suggest 1 or 2 follow-up questions grounded in the specific content you just mentioned — for example, if you referenced two projects, ask if they'd like to know more about one of them specifically. The suggestions must feel like a direct continuation of what was just said, not generic topics. Keep them short and conversational, not a formal list. "
            "Write in plain, natural prose. Do not use markdown formatting such as bold, italics, headers, or bullet points."
        ),
        "no_results": "I don't have enough information in my knowledge base to answer that. Feel free to reach out to Dr. Antúnez directly at martin.antunez.ai@gmail.com.",
        "welcome": (
            "Hey there! I'm Mary, Dr. Antúnez's AI assistant.\n\n"
            "I can tell you everything about his hands-on experience building ML pipelines, "
            "production AI systems, data engineering solutions, and MLOps infrastructure.\n\n"
            "Try asking me things like:\n"
            '  • "What production ML systems has he built?"\n'
            '  • "What cloud platforms and tools has he worked with?"\n'
            '  • "What are his MLOps and data pipeline projects?"\n'
            '  • "What is his experience with AI model deployment?"\n\n'
            "Go ahead — type your question below. Type 'quit' to exit.\n"
        ),
        "goodbye": "Goodbye! Feel free to come back anytime.",
        "quit_words": {"quit", "exit", "q"},
    },
    "es": {
        "name": "María",
        "system_instructions": (
            "Eres María, una asistente cálida y muy bien informada que representa al Dr. Antúnez en su sitio web personal. "
            "Tu propósito es ayudar a los visitantes a conocer el perfil profesional del Dr. Antúnez — sus proyectos de IA y machine learning, "
            "trabajo en ingeniería de datos, experiencia en MLOps, infraestructura cloud, habilidades técnicas y trayectoria profesional — "
            "basándote estrictamente en el contexto recuperado proporcionado. "
            "Cuando el contexto mencione varios proyectos o tecnologías relevantes, menciónalos todos — nunca solo el primero. "
            "Sé clara y directa — no sobreexpliques ni añadas detalles innecesarios. Di lo que importa y sigue adelante. "
            "Nunca añadas hechos, tecnologías, detalles de proyectos ni afirmaciones que no estén presentes en el contexto. "
            "Si el contexto no contiene suficiente información para responder, reconócelo con calidez e invita al visitante "
            "a contactar directamente en martin.antunez.ai@gmail.com. "
            "Si un visitante pregunta sobre algo no relacionado con el Dr. Antúnez, indícale con amabilidad que estás aquí para hablar "
            "sobre el perfil y trabajo del Dr. Antúnez, e invítalo a hacer una pregunta relevante. "
            "DEBES responder SIEMPRE en español, independientemente del idioma en que se haga la pregunta. "
            "Sé cálida y conversacional pero mantén las respuestas concisas. "
            "Siempre que menciones la formación académica o el background no técnico del Dr. Antúnez, DEBES cerrar con exactamente una frase original — redactada de forma distinta cada vez — que conecte brevemente esa formación con su valor concreto como científico de datos o ingeniero de MLOps. La frase debe sonar natural y adaptada al contexto específico de lo que acabas de mencionar, no una fórmula fija. Esta frase es obligatoria, no opcional. No elabores más allá de esa frase. "
            "Cuando respondas sobre formación o trayectoria, si el contexto solo muestra parte de su historial, indícalo de forma natural y deja que el visitante sepa que puede preguntar por más detalles sobre un título o período específico. "
            "Al final de cada respuesta, sugiere de forma natural 1 o 2 preguntas de seguimiento basadas en el contenido específico que acabas de mencionar — por ejemplo, si nombraste dos proyectos, pregunta si quieren saber más sobre alguno de ellos en particular. Las sugerencias deben sentirse como una continuación directa de lo que se acaba de decir, no como temas genéricos. Mantenlas breves y conversacionales, no como una lista formal. "
            "Escribe en prosa natural y clara. No uses formato markdown como negritas, cursivas, encabezados o listas con viñetas."
        ),
        "no_results": "No tengo suficiente información en mi base de conocimiento para responder eso. Puedes contactar al Dr. Antúnez directamente en martin.antunez.ai@gmail.com.",
        "welcome": (
            "¡Hola! Soy María, la asistente de IA del Dr. Antúnez.\n\n"
            "Puedo contarte todo sobre su experiencia construyendo pipelines de ML, "
            "sistemas de IA en producción, soluciones de ingeniería de datos e infraestructura MLOps.\n\n"
            "Prueba preguntándome cosas como:\n"
            '  • "¿Qué sistemas de ML en producción ha desarrollado?"\n'
            '  • "¿Con qué plataformas cloud y herramientas ha trabajado?"\n'
            '  • "¿Cuáles son sus proyectos de MLOps y pipelines de datos?"\n'
            '  • "¿Qué experiencia tiene en despliegue de modelos de IA?"\n\n'
            "Adelante — escribe tu pregunta. Escribe 'salir' para terminar.\n"
        ),
        "goodbye": "¡Hasta luego! Vuelve cuando quieras.",
        "quit_words": {"salir", "exit", "q"},
    },
}

DEFAULT_LOG_PATH = "data/logs/conversations.jsonl"


def log_entry(query: str, answer: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.temperature = temperature
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

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE) -> list[SearchResult]:
        """Return chunks most semantically similar to the query, diverse across sources."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = embed_texts([query], model=self.embedding_model)[0]
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * CANDIDATE_MULTIPLIER, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        candidates: list[SearchResult] = []
        for result_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance)
            if score >= min_score:
                candidates.append(
                    SearchResult(
                        id=result_id,
                        source=str(metadata.get("source", "unknown")),
                        text=text,
                        score=score,
                        metadata=metadata,
                    )
                )

        # enforce per-source cap to ensure results span multiple documents
        source_counts: dict[str, int] = {}
        results: list[SearchResult] = []
        for candidate in candidates:
            if source_counts.get(candidate.source, 0) < MAX_CHUNKS_PER_SOURCE:
                results.append(candidate)
                source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
            if len(results) >= top_k:
                break

        return results

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        system_instructions: str = "",
        no_results_message: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a concise answer from retrieved context."""

        if len(query) > MAX_QUERY_LENGTH:
            return (
                f"Your question is too long ({len(query)} characters). "
                f"Please keep it under {MAX_QUERY_LENGTH} characters."
            )

        results = self.retrieve(query, top_k, min_score)
        if not results and not history:
            return no_results_message or PERSONAS["en"]["no_results"]

        return self.generate_answer(query, results, system_instructions, history)

    def generate_answer(
        self,
        query: str,
        results: list[SearchResult],
        system_instructions: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Ask the chat model for a focused answer grounded in retrieved chunks."""

        require_openai_key()
        client = OpenAI()

        history_text = ""
        if history:
            turns = "\n\n".join(
                f"User: {turn['query']}\nAssistant: {turn['answer']}"
                for turn in history[-3:]
            )
            history_text = f"Previous conversation:\n{turns}\n\n"

        context_text = f"Retrieved context:\n{format_context(results)}" if results else ""
        user_input = f"{history_text}Current question: {query}\n\n{context_text}".strip()

        response = client.responses.create(
            model=self.chat_model,
            instructions=system_instructions or PERSONAS["en"]["system_instructions"],
            input=user_input,
            temperature=self.temperature,
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (0 = deterministic).")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks without generating an answer.")
    parser.add_argument("--log-file", default=DEFAULT_LOG_PATH, help="Path to JSONL conversation log. Pass empty string to disable.")
    return parser.parse_args()


def select_persona() -> dict[str, str]:
    print("\nWhich assistant would you like to talk to today? / ¿Con qué asistente te gustaría hablar hoy?\n")
    print("  1. Mary  — English")
    print("  2. María — Español\n")
    while True:
        try:
            choice = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return PERSONAS["en"]
        if choice == "2":
            return PERSONAS["es"]
        if choice in {"1", ""}:
            return PERSONAS["en"]
        print("  Please enter 1 or 2 / Por favor ingresa 1 o 2.")


def run_interactive(pipeline: RAGPipeline, top_k: int, min_score: float, show_context: bool, log_path: Path | None = None) -> None:
    persona = select_persona()
    print(f"\n{persona['welcome']}")

    name = persona["name"]
    system_instructions = persona["system_instructions"]
    no_results_message = persona["no_results"]
    quit_words = persona["quit_words"]
    goodbye = persona["goodbye"]

    session_count = 0
    history: list[dict[str, str]] = []
    recent_timestamps: deque[float] = deque()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{goodbye}")
            break

        if not query:
            continue
        if query.lower() in quit_words:
            print(goodbye)
            break

        # Session cap
        if session_count >= MAX_SESSION_QUERIES:
            print(f"\n{name}: I've reached my session limit. Please refresh to start a new conversation.")
            break

        # Rate limit: drop timestamps older than 60 seconds, then check count
        now = time.monotonic()
        while recent_timestamps and now - recent_timestamps[0] > 60:
            recent_timestamps.popleft()
        if len(recent_timestamps) >= MAX_QUERIES_PER_MINUTE:
            wait = 60 - (now - recent_timestamps[0])
            print(f"\n{name}: Just a moment — please wait {wait:.0f} seconds before the next question.")
            continue

        recent_timestamps.append(now)
        session_count += 1

        if show_context:
            results = pipeline.retrieve(query, top_k, min_score)
            output = format_context(results)
        else:
            output = pipeline.answer(query, top_k, min_score, system_instructions, no_results_message, history)
            history.append({"query": query, "answer": output})

        print(f"\n{name}: {output}\n")
        if log_path:
            log_entry(query, output, log_path)

        remaining = MAX_SESSION_QUERIES - session_count
        if remaining <= 5:
            print(f"  ({remaining} questions remaining in this session)")


def main() -> None:
    args = parse_args()
    pipeline = RAGPipeline(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        temperature=args.temperature,
    )

    log_path = Path(args.log_file) if args.log_file else None

    if args.query:
        results = pipeline.retrieve(args.query, args.top_k, args.min_score)
        if args.show_context:
            output = format_context(results)
        else:
            output = pipeline.answer(args.query, args.top_k, args.min_score)
        print(output)
        if log_path:
            log_entry(args.query, output, log_path)
    else:
        run_interactive(pipeline, args.top_k, args.min_score, args.show_context, log_path)


if __name__ == "__main__":
    main()
