# frontend/app.py
"""
Streamlit Frontend for Multi-Modal RAG System

Supports:
- PDF ingestion (text + images)
- Image folder ingestion
- CSV ingestion
- Markdown ingestion
- Audio ingestion (Whisper)
- Video ingestion (frames + transcript)
- Multimodal Q&A using Claude (Bedrock)

Key behavior (IMPORTANT):
- If the last ingestion was "Images Folder", Ask will run in "images" mode
  so retrieval is restricted to ONLY folder images.
"""

# -------------------------------------------------
# Fix Python path for Streamlit (VERY IMPORTANT)
# -------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------
# Standard imports
# -------------------------------------------------
import os
import tempfile
import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from backend.rag_service import MultiModalRAGService
from backend.vectorstore import delete_collection  # for optional isolation/reset


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def show_base64_image(img_b64: str) -> Image.Image:
    img_bytes = base64.b64decode(img_b64)
    return Image.open(BytesIO(img_bytes))


# -------------------------------------------------
# Session state defaults
# -------------------------------------------------
if "last_ingest_type" not in st.session_state:
    st.session_state.last_ingest_type = None

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "auto"  # default

if "last_image_folder" not in st.session_state:
    st.session_state.last_image_folder = ""


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal RAG Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Multi-Modal RAG Assistant")
st.caption("PDF · Images · CSV · Markdown · Audio · Video · Claude + Bedrock")

# -------------------------------------------------
# Init service (singleton per session)
# -------------------------------------------------
@st.cache_resource
def get_service():
    return MultiModalRAGService()


service = get_service()

# -------------------------------------------------
# Sidebar – Ingestion
# -------------------------------------------------
st.sidebar.header("📥 Ingest Data")

ingest_type = st.sidebar.selectbox(
    "Choose data type",
    ["PDF", "Images Folder", "CSV", "Markdown", "Audio", "Video"],
)

# ---------------- PDF ----------------
if ingest_type == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file and st.sidebar.button("Ingest PDF"):
        with st.spinner("Ingesting PDF (text + images)..."):
            path = save_uploaded_file(pdf_file)
            service.ingest_pdf(path)

        st.session_state.last_ingest_type = "PDF"
        st.session_state.active_mode = "auto"
        st.sidebar.success("✅ PDF ingested successfully")

# ---------------- Images Folder ----------------
elif ingest_type == "Images Folder":
    st.sidebar.info("Place images in a local folder and provide the folder path.")
    image_dir = st.sidebar.text_input("Image folder path", value=st.session_state.last_image_folder)

    isolate = st.sidebar.checkbox(
        "Isolate folder (clear old index)",
        value=True,
        help="Recommended: clears existing PGVector collection so ONLY folder images are used.",
    )

    if st.sidebar.button("Ingest Images"):
        if not image_dir.strip():
            st.sidebar.error("Please enter an image folder path.")
        else:
            with st.spinner("Ingesting images from folder..."):
                # OPTIONAL but strongly recommended to avoid PDF/video leakage
                if isolate:
                    from backend.vectorstore import delete_all_collections

                    delete_all_collections()


                service.ingest_images_folder(image_dir)

            st.session_state.last_ingest_type = "Images Folder"
            st.session_state.last_image_folder = image_dir
            st.session_state.active_mode = "images"  # 🔥 FORCE image-only mode
            st.sidebar.success("✅ Images ingested successfully (folder-only mode enabled)")

# ---------------- CSV ----------------
elif ingest_type == "CSV":
    csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if csv_file and st.sidebar.button("Ingest CSV"):
        with st.spinner("Ingesting CSV..."):
            path = save_uploaded_file(csv_file)
            service.ingest_csv(path)

        st.session_state.last_ingest_type = "CSV"
        st.session_state.active_mode = "auto"
        st.sidebar.success("✅ CSV ingested successfully")

# ---------------- Markdown ----------------
elif ingest_type == "Markdown":
    md_file = st.sidebar.file_uploader("Upload Markdown", type=["md"])

    if md_file and st.sidebar.button("Ingest Markdown"):
        with st.spinner("Ingesting Markdown..."):
            path = save_uploaded_file(md_file)
            service.ingest_markdown(path)

        st.session_state.last_ingest_type = "Markdown"
        st.session_state.active_mode = "auto"
        st.sidebar.success("✅ Markdown ingested successfully")

# ---------------- Audio ----------------
elif ingest_type == "Audio":
    audio_file = st.sidebar.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac"])

    if audio_file and st.sidebar.button("Ingest Audio"):
        with st.spinner("Transcribing & ingesting audio..."):
            path = save_uploaded_file(audio_file)
            service.ingest_audio(path)

        st.session_state.last_ingest_type = "Audio"
        st.session_state.active_mode = "audio"
        st.sidebar.success("✅ Audio ingested successfully")

# ---------------- Video ----------------
elif ingest_type == "Video":
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

    if video_file and st.sidebar.button("Ingest Video"):
        with st.spinner("Extracting frames & transcribing video..."):
            path = save_uploaded_file(video_file)
            service.ingest_video(path)

        st.session_state.last_ingest_type = "Video"
        st.session_state.active_mode = "video"
        st.sidebar.success("✅ Video ingested successfully")

# -------------------------------------------------
# Main – Ask Questions
# -------------------------------------------------
st.divider()
st.header("💬 Ask a Question")

# Show mode but keep it in sync with last ingest
mode_options = ["images", "auto", "audio", "video"]
default_mode = st.session_state.active_mode if st.session_state.active_mode in mode_options else "auto"
mode_index = mode_options.index(default_mode)

mode = st.selectbox(
    "Query mode",
    [
        "auto",
        "image_folder",
        "audio",
        "video",
    ],
    help="""
auto = mixed multimodal
image_folder = ONLY images from folder
audio = audio transcripts only
video = transcript + frames
""",
)

question = st.text_area(
    "Your question",
    placeholder="Ask anything about the ingested data...",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # 🔥 HARD ENFORCEMENT:
        # If last ingest was Images Folder, force 'images' mode even if user forgot to switch.
        effective_mode = mode
        if st.session_state.last_ingest_type == "Images Folder":
            effective_mode = "images"

        with st.spinner(f"Thinking... (mode: {effective_mode})"):
            result = service.ask(question, mode=effective_mode)

        st.subheader("🧠 Answer")
        st.write(result["answer"])

        # ---------------- Images ----------------
        if result.get("images"):
            st.subheader("🖼️ Relevant Visuals")
            cols = st.columns(min(3, len(result["images"])))

            for i, img_b64 in enumerate(result["images"][:6]):
                cols[i % len(cols)].image(
                    show_base64_image(img_b64),
                    caption=f"Retrieved image {i + 1}",
                    use_container_width=True,
                )

        # ---------------- Debug (optional) ----------------
        with st.expander("🔍 Retrieved Context (debug)"):
            st.write("Effective mode:", effective_mode)
            st.write("Last ingest type:", st.session_state.last_ingest_type)
            st.write("Text chunks:", len(result.get("texts", [])))
            st.write("Images:", len(result.get("images", [])))
            for t in result.get("texts", [])[:5]:
                st.markdown(f"- {t[:300]}...")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption("Powered by AWS Bedrock · Claude 3 Sonnet · Titan Embeddings · PGVector · Whisper")
