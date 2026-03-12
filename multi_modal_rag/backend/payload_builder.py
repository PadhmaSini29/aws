# backend/payload_builder.py
"""
Claude (Bedrock) multimodal payload builder.

Builds Bedrock-compatible request payloads for:
- Text-only RAG
- Image + text multimodal RAG
- Audio transcript Q&A
- Video transcript + frame Q&A

This module mirrors the notebook payload construction exactly,
but in a clean, reusable backend form.
"""

from __future__ import annotations

import json
from typing import List

from backend.config import settings


def build_bedrock_payload(
    *,
    question: str,
    texts: List[str] | None = None,
    images: List[str] | None = None,
) -> str:
    """
    Build a multimodal Bedrock payload for Claude 3 Sonnet.

    Args:
        question (str): user question
        texts (List[str] | None): retrieved text context
        images (List[str] | None): base64-encoded images (optional)

    Returns:
        str: JSON string payload for bedrock_client.invoke_model()
    """
    texts = texts or []
    images = images or []

    context_text = "\n\n".join(texts) if texts else "No text context available."

    # -------------------------
    # Build message content
    # -------------------------
    content = [
        {
            "type": "text",
            "text": f"""Answer the question based ONLY on the given context.

Question:
{question}

Context:
{context_text}
""",
        }
    ]

    # Attach images (multimodal)
    for img_b64 in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            }
        )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }

    return json.dumps(body)


# -----------------------------
# Specialized helpers
# -----------------------------
def build_audio_payload(
    *,
    question: str,
    transcripts: List[str],
) -> str:
    """
    Build a text-only payload for audio transcript Q&A.

    This enforces:
    - NO hallucination
    - Answer ONLY from transcript
    """
    context_text = "\n\n".join(transcripts) if transcripts else "No transcript available."

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Answer the question using ONLY the audio transcript below.

Question:
{question}

Transcript:
{context_text}
""",
                    }
                ],
            }
        ],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }

    return json.dumps(body)


def build_video_payload(
    *,
    question: str,
    transcript: str,
    frames_b64: List[str],
) -> str:
    """
    Build a multimodal payload for video Q&A.

    Uses:
    - Video transcript (always)
    - Most relevant frames
    """
    content = [
        {
            "type": "text",
            "text": f"""Answer the question using the provided transcript and visual evidence.

Question:
{question}

Transcript:
{transcript}
""",
        }
    ]

    for img_b64 in frames_b64:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            }
        )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }

    return json.dumps(body)


# Optional local test
if __name__ == "__main__":
    test_payload = build_bedrock_payload(
        question="What does the chart show?",
        texts=["This chart shows wildfire trends from 1990 to 2020."],
        images=[],
    )
    print("Payload built successfully")
