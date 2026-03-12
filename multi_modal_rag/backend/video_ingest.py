# backend/video_ingest.py
"""
Video ingestion pipeline.

Implements full notebook logic:
- Extract frames from video (OpenCV)
- Summarize frames using Claude 3 Sonnet (Bedrock multimodal)
- Optionally extract audio using ffmpeg
- Transcribe audio using Whisper
- Store:
    - video frames as type="video_frame" (with image_base64)
    - video transcript as type="video_transcript"
"""

from __future__ import annotations

import os
import json
import base64
import mimetypes
import subprocess
from typing import List

import cv2
import whisper
from langchain_core.documents import Document

from backend.config import settings
from backend.bedrock_client import get_bedrock_client
from backend.vectorstore import add_documents


# -----------------------------
# Helpers
# -----------------------------
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_media_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"


def _summarize_frame_with_claude(
    image_b64: str,
    media_type: str,
    prompt: str,
) -> str:
    client = get_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    response = client.invoke_model(
        modelId=settings.claude_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# -----------------------------
# Whisper model (cached)
# -----------------------------
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(settings.whisper_model_name)
    return _whisper_model


# -----------------------------
# Audio extraction
# -----------------------------
def _extract_audio(video_path: str, wav_path: str) -> None:
    """
    Extract mono 16kHz WAV audio using ffmpeg.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            wav_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# -----------------------------
# Main ingestion function
# -----------------------------
def ingest_video(
    video_path: str,
    *,
    every_seconds: int = 2,
) -> None:
    """
    Ingest a video file into the vector store.

    Args:
        video_path (str): path to video file
        every_seconds (int): frame extraction interval
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    settings.ensure_dirs()
    os.makedirs(settings.video_frames_dir, exist_ok=True)

    # -------------------------
    # 1️⃣ FRAME EXTRACTION
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frame_interval = max(int(fps * every_seconds), 1)

    frame_paths: List[str] = []
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(
                settings.video_frames_dir, f"frame_{saved:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved += 1

        count += 1

    cap.release()

    # -------------------------
    # 2️⃣ FRAME SUMMARIZATION
    # -------------------------
    frame_docs: List[Document] = []

    for frame_path in frame_paths:
        img_b64 = _encode_image(frame_path)
        media_type = _guess_media_type(frame_path)

        summary = _summarize_frame_with_claude(
            image_b64=img_b64,
            media_type=media_type,
            prompt=settings.frame_prompt,
        )

        frame_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "type": settings.type_video_frame,
                    "source": frame_path,
                    "video": video_path,
                    "image_base64": img_b64
                    if settings.store_raw_base64_in_metadata
                    else None,
                },
            )
        )

    if frame_docs:
        add_documents(
            frame_docs,
            
            collection=settings.collection_video_frames

        )


    # -------------------------
    # 3️⃣ AUDIO TRANSCRIPTION
    # -------------------------
    if settings.enable_video_audio_transcription:
        audio_wav = os.path.join(
            settings.audio_dir, "video_audio.wav"
        )

        try:
            _extract_audio(video_path, audio_wav)

            whisper_model = _get_whisper_model()
            result = whisper_model.transcribe(audio_wav)
            transcript = result.get("text", "").strip()

            if transcript:
                transcript_doc = Document(
                    page_content=transcript,
                    metadata={
                        "type": settings.type_video_transcript,
                        "source": video_path,
                    },
                )
                add_documents(
                    transcript_docs,
                    collection=settings.video_text_collection
                )


        except Exception as exc:
            print(f"⚠️ Audio transcription skipped: {exc}")

    print(
        f"✅ Video ingested | frames: {len(frame_docs)} | "
        f"audio transcript: {'yes' if settings.enable_video_audio_transcription else 'no'}"
    )


# Optional CLI test
if __name__ == "__main__":
    path = input("Enter video file path: ").strip()
    ingest_video(path)
