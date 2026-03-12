from pathlib import Path
from typing import Dict, Any
import json

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf is required to load PDF files") from e

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as e:
        raise ImportError("python-docx is required to load DOCX files") from e

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def load_document(path_str: str) -> Dict[str, Any]:
    """
    Load a document from disk and normalize it.

    Supported formats:
    - .txt
    - .json
    - .pdf
    - .docx

    Returns:
    {
      "source_path": "...",
      "doc_type": "TXT|JSON|PDF|DOCX|UNKNOWN",
      "text": "string content"
    }
    """
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = _load_txt(path)
        doc_type = "TXT"
    elif suffix == ".json":
        data = _load_json(path)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        doc_type = "JSON"
    elif suffix == ".pdf":
        text = _load_pdf(path)
        doc_type = "PDF"
    elif suffix == ".docx":
        text = _load_docx(path)
        doc_type = "DOCX"
    else:
        text = _load_txt(path)
        doc_type = "UNKNOWN"

    return {
        "source_path": str(path),
        "doc_type": doc_type,
        "text": text,
    }
