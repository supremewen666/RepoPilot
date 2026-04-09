"""Repository document loading, normalization, and chunk selection helpers."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from repopilot.compat import Document
from repopilot.rag.chunking import (
    ChunkingConfig,
    semantic_chunk,
    select_chunk_strategy,
    sliding_window_chunk,
    structured_chunk,
)
from repopilot.rag.utils import slugify

if TYPE_CHECKING:
    from repopilot.rag.easyrag import EasyRAG

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency path.
    PdfReader = None

_TEXT_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
_PDF_SUFFIXES = {".pdf"}
_EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "node_modules", ".repopilot"}
_PREFERRED_DOC_DIRS = {"docs", "doc", "design", "adr", "adrs", "notes"}
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def is_indexable_document(path: Path, repo_root: Path) -> bool:
    """Return whether a repository file should be loaded into RAG."""

    if any(part in _EXCLUDED_PARTS for part in path.parts):
        return False

    lower_name = path.name.lower()
    if lower_name.startswith("readme"):
        return True

    relative_parts = {part.lower() for part in path.relative_to(repo_root).parts}
    if path.suffix.lower() in _PDF_SUFFIXES:
        return bool(relative_parts & _PREFERRED_DOC_DIRS)
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return False
    return bool(relative_parts & _PREFERRED_DOC_DIRS) or lower_name in {
        "contributing.md",
        "architecture.md",
        "design.md",
        "roadmap.md",
        "setup.md",
        "install.md",
        "getting_started.md",
    }


def _save_pdf_page_images(page: object, path: Path, root: Path, *, page_number: int) -> list[str]:
    """Persist embedded PDF page images for later multimodal retrieval."""

    raw_images = getattr(page, "images", None)
    if not raw_images:
        return []
    try:
        images = list(raw_images)
    except TypeError:
        return []

    output_dir = root / ".repopilot" / "media" / path.stem / f"page-{page_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for index, image in enumerate(images, start=1):
        image_name = Path(str(getattr(image, "name", f"image-{index}.png"))).name or f"image-{index}.png"
        suffix = Path(image_name).suffix or ".png"
        output_path = output_dir / f"image-{index}{suffix}"

        data = getattr(image, "data", None)
        if isinstance(data, bytes) and data:
            output_path.write_bytes(data)
            saved_paths.append(str(output_path))
            continue

        pil_image = getattr(image, "image", None)
        if pil_image is None:
            continue
        buffer = io.BytesIO()
        try:
            pil_image.save(buffer, format=(pil_image.format or "PNG"))
        except Exception:
            continue
        output_path = output_path.with_suffix(f".{(pil_image.format or 'png').lower()}")
        output_path.write_bytes(buffer.getvalue())
        saved_paths.append(str(output_path))
    return saved_paths


def build_document_metadata(
    path: Path,
    root: Path,
    *,
    source_type: str,
    page_number: int | None = None,
    image_paths: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build canonical metadata for a loaded repository document."""

    relative_path = str(path.relative_to(root))
    title = path.stem
    doc_id = f"doc::{slugify(relative_path)}"
    if page_number is not None:
        doc_id = f"{doc_id}::page::{page_number}"
    metadata: dict[str, object] = {
        "source_type": source_type,
        "path": str(path),
        "relative_path": relative_path,
        "title": title,
        "doc_id": doc_id,
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    if image_paths:
        metadata["image_paths"] = list(image_paths)
        metadata["has_visual_content"] = True
    return metadata


def load_pdf_documents(path: Path, root: Path) -> list[Document]:
    """Load text pages from a PDF file."""

    if PdfReader is None:
        return []
    try:
        reader = PdfReader(str(path))
    except Exception:
        return []

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        image_paths = _save_pdf_page_images(page, path, root, page_number=page_number)
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if not text and not image_paths:
            continue
        if not text and image_paths:
            text = f"Scanned PDF page {page_number} from {path.stem}."
        documents.append(
            Document(
                page_content=text,
                metadata=build_document_metadata(path, root, source_type="pdf", page_number=page_number, image_paths=image_paths),
            )
        )
    return documents


def load_repo_documents(repo_root: str | Path) -> list[Document]:
    """Load repository documents suitable for knowledge indexing."""

    root = Path(repo_root).resolve()
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_indexable_document(path, root):
            continue
        if path.suffix.lower() in _PDF_SUFFIXES:
            documents.extend(load_pdf_documents(path, root))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata=build_document_metadata(path, root, source_type="doc"),
            )
        )
    return documents


def document_prefers_structured(document: Document) -> bool:
    """Return whether a document likely benefits from structure-aware chunking."""

    path = str(document.metadata.get("path", "")).lower()
    if path.endswith((".md", ".mdx", ".rst")):
        return True
    if path.endswith(".pdf") and _HEADING_PATTERN.search(document.page_content):
        return True
    return str(document.metadata.get("title", "")).lower().startswith("readme")


def chunk_with_strategy(
    document: Document,
    *,
    config: ChunkingConfig,
    strategy: str,
    rag: "EasyRAG" | None = None,
) -> list[Document]:
    """Run the selected chunking strategy with graceful fallback."""

    if rag is not None and strategy in rag.chunker_registry:
        chunker = rag.chunker_registry[strategy]
        try:
            if strategy == "semantic":
                chunks = chunker(document, config=config, embedding_func=rag.embedding_func)
            else:
                chunks = chunker(document, config=config)
            if chunks:
                return chunks
        except Exception:
            pass

    if strategy == "structured":
        chunks = structured_chunk(document, config=config)
        if chunks:
            return chunks
    if strategy == "semantic":
        chunks = semantic_chunk(document, config=config, embedding_func=None if rag is None else rag.embedding_func)
        if chunks:
            return chunks
    return sliding_window_chunk(document, config=config)


def chunk_documents(
    documents: list[Document],
    *,
    config: ChunkingConfig | None = None,
    chunk_strategy_override: str | None = None,
    rag: "EasyRAG" | None = None,
) -> list[Document]:
    """Split documents into primary-strategy chunks with overlap metadata."""

    chunking = config or ChunkingConfig()
    chunks: list[Document] = []
    for document in documents:
        strategy = chunk_strategy_override or select_chunk_strategy(document)
        if strategy == "semantic" and document_prefers_structured(document):
            strategy = "structured"
        chunks.extend(chunk_with_strategy(document, config=chunking, strategy=strategy, rag=rag))
    return chunks


def prepare_documents_for_insert(
    texts: str | Sequence[str],
    *,
    ids: Sequence[str] | None = None,
    file_paths: Sequence[str] | None = None,
) -> list[Document]:
    """Normalize raw insert inputs into Document objects."""

    normalized_texts = [texts] if isinstance(texts, str) else list(texts)
    normalized_ids = list(ids) if ids is not None else []
    normalized_paths = list(file_paths) if file_paths is not None else []

    documents: list[Document] = []
    for index, text in enumerate(normalized_texts):
        content = text.strip()
        if not content:
            continue
        file_path = normalized_paths[index] if index < len(normalized_paths) else ""
        title_source = Path(file_path).stem if file_path else f"Document {index + 1}"
        source_type = "pdf" if file_path.lower().endswith(".pdf") else "doc"
        document_id = normalized_ids[index] if index < len(normalized_ids) else f"doc::{slugify(file_path or f'document-{index}')}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source_type": source_type,
                    "path": file_path,
                    "relative_path": file_path,
                    "title": title_source or f"Document {index + 1}",
                    "doc_id": document_id,
                },
            )
        )
    return documents
