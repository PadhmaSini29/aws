# backend/rag_service.py
"""
High-level RAG service.

This is the orchestration layer that:
- Exposes ingestion APIs
- Routes retrieval by mode
- Builds Bedrock payloads
- Invokes Claude on Bedrock
- Returns final answer + retrieved visuals
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from backend.config import settings
from backend.bedrock_client import get_bedrock_client

# Ingestion
from backend.pdf_ingest import ingest_pdf
from backend.image_ingest import ingest_images_folder
from backend.csv_ingest import ingest_csv
from backend.markdown_ingest import ingest_markdown
from backend.audio_ingest import ingest_audio
from backend.video_ingest import ingest_video

# Retrieval
from backend.retrieval import (
    retrieve_context,
    retrieve_image_folder_only,
    retrieve_audio_only,
    retrieve_video_only,
)

# Payload builders
from backend.payload_builder import (
    build_bedrock_payload,
    build_audio_payload,
    build_video_payload,
)

# Vectorstore bootstrap
from backend.vectorstore import bootstrap_all_collections



class MultiModalRAGService:
    """
    Main service class for Multi-Modal RAG.

    Used by:
    - Streamlit UI
    - APIs
    - CLI / batch jobs
    """

    def __init__(self):
        bootstrap_all_collections()
        self.client = get_bedrock_client()


    # =================================================
    # INGESTION APIS
    # =================================================
    def ingest_pdf(self, pdf_path: str) -> None:
        ingest_pdf(pdf_path)

    def ingest_images_folder(self, image_dir: str) -> None:
        ingest_images_folder(image_dir)

    def ingest_csv(self, csv_path: str) -> None:
        ingest_csv(csv_path)

    def ingest_markdown(self, md_path: str) -> None:
        ingest_markdown(md_path)

    def ingest_audio(self, audio_path: str) -> None:
        ingest_audio(audio_path)

    def ingest_video(self, video_path: str) -> None:
        ingest_video(video_path)

    # =================================================
    # QUESTION ANSWERING
    # =================================================
    def ask(
        self,
        question: str,
        *,
        k: Optional[int] = None,
        mode: str = "auto",
    ) -> Dict[str, object]:
        """
        Ask a question using RAG.

        Modes:
        - auto          → mixed multimodal RAG
        - image_folder  → ONLY images from image folder
        - audio         → audio transcripts only
        - video         → video transcript + frames
        """
        if not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "texts": [],
                "images": [],
            }

        k = k or settings.top_k
        mode = mode.lower()

        # -----------------------------
        # IMAGE-FOLDER-ONLY MODE
        # -----------------------------
        if mode == "image_folder":
            context = retrieve_image_folder_only(question, k=k)

            payload = build_bedrock_payload(
                question=question,
                texts=context["texts"],
                images=context["images"],
            )
        
        # -----------------------------
        # AUDIO-ONLY MODE
        # -----------------------------
        elif mode == "audio":
            context = retrieve_audio_only(question, k=k)

            payload = build_audio_payload(
                question=question,
                transcripts=context["texts"],
            )

        # -----------------------------
        # VIDEO MODE
        # -----------------------------
        elif mode == "video":
            context = retrieve_video_only(question, k=k)

            transcript_text = "\n\n".join(context["texts"]) if context["texts"] else ""

            payload = build_video_payload(
                question=question,
                transcript=transcript_text,
                frames_b64=context["images"],
            )

        # -----------------------------
        # AUTO / MULTIMODAL MODE
        # -----------------------------
        else:
            context = retrieve_context(
                question,
                k=k,
            )

            payload = build_bedrock_payload(
                question=question,
                texts=context["texts"],
                images=context["images"],
            )

        # -----------------------------
        # Invoke Claude (Bedrock)
        # -----------------------------
        response = self.client.invoke_model(
            modelId=settings.claude_model_id,
            body=payload,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        answer = response_body["content"][0]["text"]

        return {
            "answer": answer,
            "texts": context.get("texts", []),
            "images": context.get("images", []),
        }


# =================================================
# CLI TEST
# =================================================
if __name__ == "__main__":
    service = MultiModalRAGService()
    q = input("Ask a question: ").strip()
    result = service.ask(q)
    print("\n===== ANSWER =====\n")
    print(result["answer"])
