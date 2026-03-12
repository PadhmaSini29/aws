# backend/config.py
"""
Central configuration for the Multi-Modal RAG project.

Loads environment variables (optionally from a .env file) and provides:
- AWS Bedrock config
- PGVector config (MULTI-COLLECTION: separate vectorstores per modality)
- Model IDs
- Chunking params
- Prompts (image/table/frame)
- Local data paths

Usage:
    from backend.config import settings
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# -------------------------------------------------
# Env helpers
# -------------------------------------------------
def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _str_env(name: str, default: str) -> str:
    val = os.getenv(name)
    return default if val is None or str(val).strip() == "" else val.strip()


def _resolve_project_root() -> Path:
    # backend/config.py -> backend -> project root
    return Path(__file__).resolve().parents[1]


# Load .env (if present) from project root
PROJECT_ROOT = _resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env", override=False)


# -------------------------------------------------
# Prompts
# -------------------------------------------------
DEFAULT_IMAGE_SUMMARY_PROMPT = """
Summarize the key information shown in this image.

If the image contains a chart, graph, or diagram:
- Describe what is being measured
- Identify the time period (if present)
- Explain major trends, patterns, or changes

If the image is not a chart:
- Describe the main visual elements
- Highlight any important details or insights
""".strip()

DEFAULT_TABLE_PROMPT = """
Extract the table from this image into structured text.

If a table is present:
- List column names
- Include units if shown
- Extract key rows or values
- Summarize notable trends or patterns

If no table is present:
- State that no table was detected
""".strip()

DEFAULT_FRAME_PROMPT = """
You are analyzing a single frame extracted from a video.

1. Describe what is visually happening in the frame.
2. Identify people, objects, screens, or scenes present.
3. If there is any on-screen text, read and summarize it accurately.
4. If the frame shows a chart, graph, or data visualization:
   - Describe the type of chart
   - Mention key values, trends, or changes
5. If this looks like part of a presentation, demo, or tutorial:
   - Explain what step or concept is being shown
6. Summarize the key information conveyed by this frame in 2–4 concise sentences.
""".strip()


# -------------------------------------------------
# Settings Dataclass
# -------------------------------------------------
@dataclass(frozen=True)
class Settings:
    # ---- PGVector collections (ADD THESE) ----
    collection_text: str = "mm_text"
    collection_images: str = "mm_images"
    collection_pdf_images: str = "mm_pdf_images"
    collection_audio: str = "mm_audio"
    collection_video_text: str = "mm_video_text"
    collection_video_frames: str = "mm_video_frames"

    # ---- Project paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdf"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    csv_dir: Path = PROJECT_ROOT / "data" / "csv"
    markdown_dir: Path = PROJECT_ROOT / "data" / "markdown"
    audio_dir: Path = PROJECT_ROOT / "data" / "audio"
    video_dir: Path = PROJECT_ROOT / "data" / "video"
    video_frames_dir: Path = PROJECT_ROOT / "video_frames"

    # ---- AWS / Bedrock
    aws_region: str = _str_env("AWS_REGION", "us-east-1")
    claude_model_id: str = _str_env(
        "CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    embed_model_id: str = _str_env("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")

    bedrock_connect_timeout: int = _int_env("BEDROCK_CONNECT_TIMEOUT", 30)
    bedrock_read_timeout: int = _int_env("BEDROCK_READ_TIMEOUT", 120)
    bedrock_max_attempts: int = _int_env("BEDROCK_MAX_ATTEMPTS", 3)

    # ---- Vector store (PGVector)
    pg_conn_string: str = _str_env(
        "PG_CONN_STRING",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/multimodal_rag",
    )

    # -------------------------------------------------
    # MULTI-COLLECTION DESIGN (🔥 REQUIRED)
    # Each modality gets its own collection to avoid leakage.
    # You can override names in .env if needed.
    # -------------------------------------------------
    collection_text: str = _str_env("COLLECTION_TEXT", "mm_text")
    collection_images: str = _str_env("COLLECTION_IMAGES", "mm_images")
    collection_pdf_images: str = _str_env("COLLECTION_PDF_IMAGES", "mm_pdf_images")
    collection_audio: str = _str_env("COLLECTION_AUDIO", "mm_audio")
    collection_video_text: str = _str_env("COLLECTION_VIDEO_TEXT", "mm_video_text")
    collection_video_frames: str = _str_env("COLLECTION_VIDEO_FRAMES", "mm_video_frames")

    # ---- Chunking / Retrieval
    chunk_size_tokens: int = _int_env("CHUNK_SIZE_TOKENS", 300)
    chunk_overlap_tokens: int = _int_env("CHUNK_OVERLAP_TOKENS", 50)
    top_k: int = _int_env("TOP_K", 3)
    max_return_images: int = 2

    # ---- Claude generation controls
    max_tokens: int = _int_env("CLAUDE_MAX_TOKENS", 1024)
    temperature: float = _float_env("CLAUDE_TEMPERATURE", 0.1)
    top_p: float = _float_env("CLAUDE_TOP_P", 0.1)

    # ---- Ingestion behavior
    store_raw_base64_in_metadata: bool = _bool_env("STORE_RAW_BASE64_IN_METADATA", True)
    enable_video_audio_transcription: bool = _bool_env(
        "ENABLE_VIDEO_AUDIO_TRANSCRIPTION", True
    )

    # ---- Whisper
    whisper_model_name: str = _str_env("WHISPER_MODEL_NAME", "base")

    # ---- Prompts
    image_summary_prompt: str = _str_env("IMAGE_SUMMARY_PROMPT", DEFAULT_IMAGE_SUMMARY_PROMPT)
    table_prompt: str = _str_env("TABLE_PROMPT", DEFAULT_TABLE_PROMPT)
    frame_prompt: str = _str_env("FRAME_PROMPT", DEFAULT_FRAME_PROMPT)

    # ---- Types (metadata tags)
    type_text: str = "text"
    type_image: str = "image"
    type_image_table: str = "image_table"
    type_csv: str = "csv"
    type_markdown: str = "markdown"
    type_audio: str = "audio"
    type_audio_transcript: str = "audio_transcript"
    type_video_frame: str = "video_frame"
    type_video_transcript: str = "video_transcript"

    def ensure_dirs(self) -> None:
        """Create required directories. Safe to call at startup."""
        for p in [
            self.data_dir,
            self.pdf_dir,
            self.images_dir,
            self.csv_dir,
            self.markdown_dir,
            self.audio_dir,
            self.video_dir,
            self.video_frames_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate critical config early to fail fast."""
        if not self.aws_region:
            raise ValueError("AWS_REGION is empty.")
        if not self.pg_conn_string:
            raise ValueError("PG_CONN_STRING is empty.")
        if not self.claude_model_id:
            raise ValueError("CLAUDE_MODEL_ID is empty.")
        if not self.embed_model_id:
            raise ValueError("EMBED_MODEL_ID is empty.")
        if self.chunk_size_tokens <= 0:
            raise ValueError("CHUNK_SIZE_TOKENS must be > 0.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be > 0.")

        # collections must exist as non-empty strings
        for name in [
            self.collection_text,
            self.collection_images,
            self.collection_pdf_images,
            self.collection_audio,
            self.collection_video_text,
            self.collection_video_frames,
        ]:
            if not name or not str(name).strip():
                raise ValueError("One of the COLLECTION_* values is empty.")


# Singleton settings object used across backend
settings = Settings()
settings.ensure_dirs()
settings.validate()
