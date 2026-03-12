# backend.py
# ============================================================
# Multimodal RAG Backend (Text + Image + Audio + Video)
# ============================================================

import os
import json
import base64
import mimetypes
import subprocess

import boto3
from botocore.config import Config

import pandas as pd
import cv2
import whisper

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document

# ============================================================
# CONFIG
# ============================================================

AWS_REGION = "us-east-1"

CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"

PG_CONN_STRING = "postgresql+psycopg2://postgres:12345678@localhost:5432/multimodal_rag"
COLLECTION_NAME = "multimodal_rag_prod"

UPLOAD_DIR = "uploads"
FRAME_DIR = "video_frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ============================================================
# BEDROCK CLIENT
# ============================================================

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        read_timeout=120,
        connect_timeout=120,
        retries={"max_attempts": 2}
    )
)

# ============================================================
# EMBEDDINGS + VECTORSTORE
# ============================================================

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id=EMBED_MODEL_ID,
    normalize=True,
)

vectorstore = PGVector(
    connection_string=PG_CONN_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# ============================================================
# HELPERS
# ============================================================

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_media_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"


def summarize_image_with_claude(img_b64: str, media_type: str, prompt: str) -> str:
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
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    return json.loads(response["body"].read())["content"][0]["text"]

# ============================================================
# INGESTION FUNCTIONS
# ============================================================

def ingest_pdf(pdf_path: str):
    # ---- TEXT ----
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=50)
    text_docs = splitter.split_documents(pages)

    vectorstore.add_documents([
        Document(d.page_content, metadata={"type": "text"})
        for d in text_docs
    ])

    # ---- IMAGES (charts/graphs) ----
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        for img in page.images:
            img_path = os.path.join(UPLOAD_DIR, img.name)
            with open(img_path, "wb") as f:
                f.write(img.data)

            img_b64 = encode_image(img_path)
            summary = summarize_image_with_claude(
                img_b64,
                guess_media_type(img_path),
                "Summarize the chart or graph and explain key trends."
            )

            vectorstore.add_documents([
                Document(
                    page_content=summary,
                    metadata={
                        "type": "image",
                        "image_base64": img_b64,
                        "source": pdf_path,
                    }
                )
            ])


def ingest_images(image_paths):
    for p in image_paths:
        img_b64 = encode_image(p)
        summary = summarize_image_with_claude(
            img_b64,
            guess_media_type(p),
            "Describe the image and summarize key visual information."
        )

        vectorstore.add_documents([
            Document(
                page_content=summary,
                metadata={"type": "image", "image_base64": img_b64}
            )
        ])


def ingest_csv(csv_path):
    df = pd.read_csv(csv_path)

    csv_text = df.to_string(index=False)

    vectorstore.add_documents([
        Document(
            page_content=csv_text,
            metadata={
                "type": "csv",
                "source": os.path.basename(csv_path)
            }
        )
    ])


def ingest_markdown(md_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        vectorstore.add_documents([
            Document(
                page_content=f.read(),
                metadata={"type": "markdown", "source": md_path}
            )
        ])


def ingest_audio(audio_path: str):
    model = whisper.load_model("base")
    transcript = model.transcribe(audio_path)["text"].strip()

    vectorstore.add_documents([
        Document(
            page_content=transcript,
            metadata={"type": "audio_transcript", "source": audio_path}
        )
    ])


def ingest_video(video_path: str):
    # ---- AUDIO ----
    audio_path = os.path.join(UPLOAD_DIR, "video_audio.wav")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ],
        check=True
    )

    ingest_audio(audio_path)

    # ---- FRAMES ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = int(fps * 2)

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame_path = os.path.join(FRAME_DIR, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            img_b64 = encode_image(frame_path)
            summary = summarize_image_with_claude(
                img_b64,
                "image/jpeg",
                "Describe what is happening in this video frame."
            )

            vectorstore.add_documents([
                Document(
                    page_content=summary,
                    metadata={
                        "type": "video_frame",
                        "image_base64": img_b64,
                        "source": video_path
                    }
                )
            ])
            saved += 1

        count += 1

    cap.release()

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(query: str, k: int = 6):
    docs = vectorstore.similarity_search(query, k=k)

    texts, images = [], []

    for d in docs:
        doc_type = d.metadata.get("type")
        if doc_type in ("image", "video_frame"):
            images.append(d.metadata.get("image_base64"))
            texts.append(d.page_content)
        else:
            texts.append(d.page_content)

    return {"texts": texts, "images": images}

# ============================================================
# CLAUDE INVOCATION
# ============================================================

def build_bedrock_payload(question: str, texts, images):
    content = [
        {
            "type": "text",
            "text": f"""You are given structured tabular data (such as CSV rows) and text.

Rules:
- Use ONLY the provided context.
- If numeric values are present, perform calculations explicitly.
- Do NOT say data is missing if values are clearly present.
- Show calculation steps when percentages or comparisons are asked.

Question:
{question}

Context:
{chr(10).join(texts[:5])}

"""

        }
    ]

    if images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": images[0]
            }
        })

    return json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
    })


def invoke_claude(payload: str):
    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=payload,
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(response["body"].read())
    return body["content"][0]["text"]

# ============================================================
# PUBLIC API (USED BY STREAMLIT)
# ============================================================

def answer_query(query: str, k: int = 6):
    context = retrieve_context(query, k=k)

    payload = build_bedrock_payload(
        question=query,
        texts=context["texts"],
        images=context["images"]
    )

    answer = invoke_claude(payload)

    return {
        "answer": answer,
        "texts": context["texts"],
        "images": context["images"]
    }

# ============================================================
# END OF FILE
# ============================================================
