"""Build a local Chroma vector store from source documents.

Change notes:
- PDF support was added so articles can be indexed directly instead of first
  converting every file to .txt.
- Recursive chunking replaced fixed-size slicing because articles often have
  useful structure: sections, paragraphs, and sentences should stay together
  when possible.
- OpenAI semantic embeddings replaced the earlier hash-based vectors. This
  makes retrieval based on meaning, not just word overlap.
- Chroma replaced the JSON vector store because it keeps embeddings, text, and
  metadata together in a local database folder that is easier to query and grow.
- Secrets stay outside the code. Put OPENAI_API_KEY in .env or your shell
  environment; never commit the real key to GitHub.

This module reads .txt and .pdf documents, splits them into overlapping chunks,
embeds them with OpenAI, and stores them in a local Chroma database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHROMA_PATH = "data/index/chroma_db"
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_PREVIEW_PATH = "data/index/chunks_preview.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)
REPEATED_LINE_MAX_LENGTH = 180
MOJIBAKE_REPLACEMENTS = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufffd": "",
    "â€œ": '"',
    "â€\u009d": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€\u0093": "-",
    "â€\u0094": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
}


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text from one source unit."""

    source: str
    text: str
    page: int = 0
    document_type: str = "text"


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of a source document."""

    id: str
    source: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


def require_openai_key() -> None:
    """Load .env and fail early if the API key is missing."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to a local .env file or set it in your shell. "
            "Do not commit the real key to GitHub."
        )


def embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64) -> list[list[float]]:
    """Create OpenAI semantic embeddings for a list of texts."""

    require_openai_key()
    client = OpenAI()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))

    return embeddings


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text recursively, preserving natural boundaries where possible."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    pieces = recursive_split(normalized, chunk_size)
    return merge_pieces_with_overlap(pieces, chunk_size, chunk_overlap)


def normalize_text(text: str) -> str:
    """Clean common extraction artifacts while preserving paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_text_artifacts(text: str) -> str:
    """Repair common encoding artifacts and drop unrecoverable replacement marks."""

    try:
        from ftfy import fix_text
    except ImportError:
        fix_text = None

    if fix_text:
        text = fix_text(text)

    for old, new in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(old, new)

    return text


def clean_pdf_pages(pages: list[DocumentPage], remove_references: bool = True) -> list[DocumentPage]:
    """Clean repeated PDF artifacts across pages before chunking."""

    if not pages:
        return []

    lightly_cleaned = [
        DocumentPage(
            source=page.source,
            text=clean_pdf_page_text(page.text),
            page=page.page,
            document_type=page.document_type,
        )
        for page in pages
    ]
    repeated_lines = find_repeated_pdf_lines(lightly_cleaned)
    cleaned_pages: list[DocumentPage] = []
    references_started = False

    for page in lightly_cleaned:
        lines: list[str] = []
        for line in page.text.splitlines():
            stripped = line.strip()
            normalized_line = normalize_repeated_line(stripped)
            if not stripped:
                lines.append("")
                continue
            if normalized_line in repeated_lines:
                continue
            if PAGE_NUMBER_RE.match(stripped):
                continue
            if remove_references and REFERENCE_HEADING_RE.match(stripped):
                references_started = True
                break
            if references_started:
                continue
            lines.append(stripped)

        if references_started and not lines:
            text = ""
        else:
            text = normalize_text("\n".join(lines))

        cleaned_pages.append(
            DocumentPage(
                source=page.source,
                text=text,
                page=page.page,
                document_type=page.document_type,
            )
        )

    return cleaned_pages


def clean_pdf_page_text(text: str) -> str:
    """Clean single-page PDF extraction artifacts that do not need page context."""

    text = repair_text_artifacts(text)
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_repeated_pdf_lines(pages: list[DocumentPage]) -> set[str]:
    """Find likely headers/footers repeated across multiple PDF pages."""

    if len(pages) < 3:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        seen_on_page: set[str] = set()
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        candidates = lines[:6] + lines[-6:]
        for line in candidates:
            normalized = normalize_repeated_line(line)
            if not normalized:
                continue
            seen_on_page.add(normalized)
        for normalized in seen_on_page:
            counts[normalized] = counts.get(normalized, 0) + 1

    threshold = max(2, int(len(pages) * 0.3))
    return {line for line, count in counts.items() if count >= threshold}


def normalize_repeated_line(line: str) -> str:
    """Normalize a line so repeated headers with page numbers still match."""

    line = line.strip().lower()
    if not line or len(line) > REPEATED_LINE_MAX_LENGTH:
        return ""
    line = re.sub(r"\d+", "#", line)
    line = re.sub(r"\s+", " ", line)
    line = line.strip(" -–—|")
    if len(line) < 4:
        return ""
    return line


def recursive_split(text: str, chunk_size: int) -> list[str]:
    """Split text from coarse to fine boundaries until pieces fit."""

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    for separator in ("\n\n", "\n"):
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) > 1:
            return split_parts(parts, chunk_size)

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) > 1:
        return split_parts(sentences, chunk_size)

    return hard_split(text, chunk_size)


def split_parts(parts: list[str], chunk_size: int) -> list[str]:
    """Recursively split parts that are still too large."""

    output: list[str] = []
    for part in parts:
        output.extend(recursive_split(part, chunk_size))
    return output


def hard_split(text: str, chunk_size: int) -> list[str]:
    """Last-resort split for text without useful boundaries."""

    return [text[start : start + chunk_size].strip() for start in range(0, len(text), chunk_size)]


def merge_pieces_with_overlap(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge small pieces into chunks and add overlap between chunks."""

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        current = piece

    if current:
        chunks.append(current.strip())

    if chunk_overlap == 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-chunk_overlap:].strip()
        overlapped.append(f"{prefix}\n\n{chunk}".strip() if prefix else chunk)

    return overlapped


def iter_source_files(data_dir: Path) -> Iterable[Path]:
    """Yield supported source files in a stable order."""

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_document_pages(path: Path, data_dir: Path) -> list[DocumentPage]:
    """Read text from a supported document path, preserving page metadata for PDFs."""

    source = repair_text_artifacts(path.relative_to(data_dir).as_posix())
    extension = path.suffix.lower()
    if extension == ".txt":
        return [
            DocumentPage(
                source=source,
                text=path.read_text(encoding="utf-8"),
                page=0,
                document_type="text",
            )
        ]
    if extension == ".pdf":
        return read_pdf_pages(path, source)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def read_pdf_pages(path: Path, source: str) -> list[DocumentPage]:
    """Extract text from a PDF using pypdf."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF ingestion requires `pypdf`. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(
            DocumentPage(
                source=source,
                text=page.extract_text() or "",
                page=index,
                document_type="pdf",
            )
        )
    return pages


def guess_title(text: str, source: str) -> str:
    """Guess a document title from the first meaningful line."""

    for line in normalize_text(text).splitlines():
        candidate = line.strip()
        if 20 <= len(candidate) <= 180 and not candidate.lower().startswith(("available online", "http")):
            return candidate
    return Path(source).stem


def guess_year(text: str, source: str) -> int:
    """Guess a publication year from text or filename."""

    match = YEAR_RE.search(f"{source}\n{text}")
    return int(match.group(0)) if match else 0


def build_chunks(
    data_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    clean_pdfs: bool = True,
    remove_references: bool = True,
) -> list[Chunk]:
    """Read supported documents below data_dir and return chunks with metadata."""

    chunks: list[Chunk] = []
    for path in iter_source_files(data_dir):
        pages = read_document_pages(path, data_dir)
        if clean_pdfs and path.suffix.lower() == ".pdf":
            pages = clean_pdf_pages(pages, remove_references=remove_references)
        first_page_text = pages[0].text if pages else ""
        title = guess_title(first_page_text, path.relative_to(data_dir).as_posix())
        year = guess_year(first_page_text, path.name)

        for page in pages:
            for index, chunk_text in enumerate(split_text(page.text, chunk_size, chunk_overlap)):
                chunk_id = f"{page.source}#p{page.page}#c{index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=page.source,
                        text=chunk_text,
                        metadata={
                            "source": page.source,
                            "page": page.page,
                            "chunk_index": index,
                            "document_type": page.document_type,
                            "title": title,
                            "year": year,
                        },
                    )
                )
    return chunks


def get_collection(chroma_path: Path | str, collection_name: str = DEFAULT_COLLECTION_NAME):
    """Open or create the local Chroma collection."""

    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def save_to_chroma(
    chunks: list[Chunk],
    chroma_path: Path | str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reset: bool = True,
) -> None:
    """Embed chunks and persist them in Chroma."""

    chroma_dir = Path(chroma_path)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )


def ingest(
    data_dir: Path | str = "data",
    chroma_path: Path | str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reset: bool = True,
    clean_pdfs: bool = True,
    remove_references: bool = True,
) -> list[Chunk]:
    """Build and save the Chroma vector store, returning chunks for inspection."""

    require_openai_key()
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    chunks = build_chunks(data_path, chunk_size, chunk_overlap, clean_pdfs, remove_references)
    save_to_chroma(chunks, chroma_path, collection_name, embedding_model, reset)
    return chunks


def save_chunk_preview(chunks: list[Chunk], output_path: Path | str = DEFAULT_PREVIEW_PATH) -> None:
    """Save chunks and metadata without embeddings so they can be inspected offline."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": chunk.id,
                "source": chunk.source,
                "text_length": len(chunk.text),
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a semantic local RAG index with Chroma.")
    parser.add_argument("--data-dir", default="data/documents", help="Directory containing .txt or .pdf files.")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Local Chroma database folder.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--append", action="store_true", help="Append to the collection instead of recreating it.")
    parser.add_argument("--dry-run", action="store_true", help="Write a chunk preview JSON without OpenAI or Chroma.")
    parser.add_argument("--preview-output", default=DEFAULT_PREVIEW_PATH, help="Path for --dry-run preview JSON.")
    parser.add_argument("--no-clean-pdfs", action="store_true", help="Disable PDF cleanup before chunking.")
    parser.add_argument("--keep-references", action="store_true", help="Keep reference sections in PDF chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        chunks = build_chunks(
            data_dir=Path(args.data_dir),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            clean_pdfs=not args.no_clean_pdfs,
            remove_references=not args.keep_references,
        )
        save_chunk_preview(chunks, args.preview_output)
        print(f"Prepared {len(chunks)} chunks without embeddings at {args.preview_output}")
        return

    chunks = ingest(
        data_dir=args.data_dir,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        reset=not args.append,
        clean_pdfs=not args.no_clean_pdfs,
        remove_references=not args.keep_references,
    )
    print(f"Indexed {len(chunks)} chunks into Chroma at {args.chroma_path}")


if __name__ == "__main__":
    main()
