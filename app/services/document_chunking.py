from __future__ import annotations

import html
import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


SENTENCE_RE = re.compile(r"(?<=[。！？.!?；;])\s*")
SPACE_RE = re.compile(r"\s+")
WORD_XML_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative")


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool = False
    engine: str = "auto"
    dpi: int = 180
    max_pages: int | None = None
    min_extracted_chars: int = 1000
    force: bool = False
    show_progress: bool = False
    progress_every: int = 1

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.max_pages is not None and self.max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if self.min_extracted_chars < 0:
            raise ValueError("min_extracted_chars must be non-negative")
        if self.progress_every <= 0:
            raise ValueError("progress_every must be positive")


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    source_type: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, str | int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag.lower() in {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_text(html.unescape(" ".join(self._parts)))


def load_document_text(source: str | Path, ocr_config: OcrConfig | None = None) -> tuple[str, str]:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        return load_web_text(source_text), "web"

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_text(path, ocr_config=ocr_config), "pdf"
    if suffix == ".docx":
        return load_docx_text(path), "docx"
    if suffix in {".html", ".htm"}:
        return strip_html(path.read_text(encoding="utf-8", errors="ignore")), "html"
    if suffix in {".txt", ".md"}:
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore")), suffix.lstrip(".")
    raise ValueError(f"Unsupported document type: {path.suffix or source_text}")


def load_pdf_text(path: Path, ocr_config: OcrConfig | None = None) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        if ocr_config is not None and ocr_config.enabled:
            return load_pdf_text_with_ocr(path, ocr_config)
        raise RuntimeError("PDF parsing requires pypdf. Install requirements.txt first.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {page_number}]\n{text}")
    extracted = normalize_text("\n\n".join(pages))
    if should_use_ocr(extracted, ocr_config):
        return load_pdf_text_with_ocr(path, ocr_config or OcrConfig(enabled=True))
    return extracted


def should_use_ocr(extracted: str, ocr_config: OcrConfig | None) -> bool:
    if ocr_config is None or not ocr_config.enabled:
        return False
    if ocr_config.force:
        return True
    if looks_like_archive_metadata(extracted):
        return True
    return len(extracted) < ocr_config.min_extracted_chars


def looks_like_archive_metadata(extracted: str) -> bool:
    compact = extracted.lower()
    markers = (
        "document generated by anna",
        "pdg_main_pages_found",
        "after_pdg2pic_conversion",
        "pdf_generation_missing_pages",
    )
    return sum(marker in compact for marker in markers) >= 2


def load_pdf_text_with_ocr(path: Path, ocr_config: OcrConfig) -> str:
    try:
        import fitz
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "PDF OCR requires PyMuPDF and Pillow. Install them with: "
            "python -m pip install PyMuPDF Pillow"
        ) from exc

    ocr = build_ocr_engine(ocr_config.engine)
    document = fitz.open(str(path))
    page_count = len(document)
    max_pages = min(page_count, ocr_config.max_pages or page_count)
    zoom = ocr_config.dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    try:
        for page_index in range(max_pages):
            if ocr_config.show_progress and (
                page_index == 0
                or page_index + 1 == max_pages
                or (page_index + 1) % ocr_config.progress_every == 0
            ):
                percent = ((page_index + 1) / max_pages) * 100
                print(
                    f"OCR progress: page {page_index + 1}/{max_pages} ({percent:.1f}%)",
                    file=sys.stderr,
                    flush=True,
                )
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = normalize_text(ocr.recognize(image))
            if text:
                pages.append(f"[page {page_index + 1}]\n{text}")
    finally:
        document.close()
    return normalize_text("\n\n".join(pages))


class OcrEngine:
    def recognize(self, image: Any) -> str:
        raise NotImplementedError


class RapidOcrEngine(OcrEngine):
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            try:
                from rapidocr import RapidOCR
            except Exception as exc:
                raise RuntimeError(
                    "RapidOCR is not installed. Install rapidocr-onnxruntime or rapidocr to OCR scanned PDFs."
                ) from exc
        self._ocr = RapidOCR()

    def recognize(self, image: Any) -> str:
        result, _ = self._ocr(image)
        if not result:
            return ""
        lines = []
        for row in result:
            if len(row) >= 2 and row[1]:
                lines.append(str(row[1]))
        return "\n".join(lines)


class TesseractOcrEngine(OcrEngine):
    def __init__(self) -> None:
        try:
            import pytesseract
        except Exception as exc:
            raise RuntimeError("pytesseract is not installed.") from exc
        self._pytesseract = pytesseract

    def recognize(self, image: Any) -> str:
        return self._pytesseract.image_to_string(image, lang="chi_sim+eng")


def build_ocr_engine(engine: str) -> OcrEngine:
    normalized = engine.strip().lower()
    errors = []
    if normalized in {"auto", "rapidocr"}:
        try:
            return RapidOcrEngine()
        except RuntimeError as exc:
            errors.append(str(exc))
            if normalized == "rapidocr":
                raise
    if normalized in {"auto", "tesseract"}:
        try:
            return TesseractOcrEngine()
        except RuntimeError as exc:
            errors.append(str(exc))
            if normalized == "tesseract":
                raise
    raise RuntimeError("No OCR engine is available. " + " ".join(errors))


def load_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = []
    for paragraph in root.iter(f"{WORD_XML_NS}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{WORD_XML_NS}t")]
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return normalize_text("\n\n".join(paragraphs))


def load_web_text(url: str, timeout: float = 15.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "SmartRecipe-RAG/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return strip_html(raw.decode(charset, errors="ignore"))


def strip_html(raw_html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()
    return parser.text()


def normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        compact = SPACE_RE.sub(" ", line).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines)


def chunk_text(
    text: str,
    source: str,
    source_type: str,
    config: ChunkingConfig | None = None,
    metadata: dict[str, str | int] | None = None,
) -> list[DocumentChunk]:
    config = config or ChunkingConfig()
    normalized = normalize_text(text)
    if not normalized:
        return []

    units = split_semantic_units(normalized, config.chunk_size)
    chunks: list[DocumentChunk] = []
    current = ""
    current_start = 0
    cursor = 0

    for unit in units:
        unit_start = normalized.find(unit, cursor)
        if unit_start < 0:
            unit_start = cursor
        proposed = join_chunk_parts(current, unit)
        if current and len(proposed) > config.chunk_size:
            chunks.append(make_chunk(chunks, source, source_type, current, current_start, metadata))
            overlap = current[-config.chunk_overlap :] if config.chunk_overlap else ""
            current = join_chunk_parts(overlap, unit)
            current_start = max(unit_start - len(overlap), 0)
        else:
            if not current:
                current_start = unit_start
            current = proposed
        cursor = unit_start + len(unit)

    if current:
        if chunks and len(current) < config.min_chunk_size:
            previous = chunks.pop()
            merged = join_chunk_parts(previous.text, current)
            chunks.append(
                make_chunk(
                    chunks,
                    source,
                    source_type,
                    merged,
                    previous.start_char,
                    metadata,
                )
            )
        else:
            chunks.append(make_chunk(chunks, source, source_type, current, current_start, metadata))
    return chunks


def split_semantic_units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.extend(part for part in SENTENCE_RE.split(paragraph) if part)
            continue
        for start in range(0, len(paragraph), chunk_size):
            units.append(paragraph[start : start + chunk_size])
    return units


def join_chunk_parts(left: str, right: str) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    return f"{left.strip()}\n{right.strip()}".strip()


def make_chunk(
    chunks: list[DocumentChunk],
    source: str,
    source_type: str,
    text: str,
    start_char: int,
    metadata: dict[str, str | int] | None,
) -> DocumentChunk:
    clean = text.strip()
    return DocumentChunk(
        chunk_id=f"{Path(source).stem or 'web'}-{len(chunks):04d}",
        source=source,
        source_type=source_type,
        text=clean,
        start_char=start_char,
        end_char=start_char + len(clean),
        metadata=dict(metadata or {}),
    )


def chunk_document(
    source: str | Path,
    config: ChunkingConfig | None = None,
    metadata: dict[str, str | int] | None = None,
    ocr_config: OcrConfig | None = None,
) -> list[DocumentChunk]:
    text, source_type = load_document_text(source, ocr_config=ocr_config)
    return chunk_text(text, str(source), source_type, config=config, metadata=metadata)


def chunk_documents(
    sources: Iterable[str | Path],
    config: ChunkingConfig | None = None,
    ocr_config: OcrConfig | None = None,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source in sources:
        chunks.extend(chunk_document(source, config=config, ocr_config=ocr_config))
    return chunks


def write_chunks_jsonl(chunks: Iterable[DocumentChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
