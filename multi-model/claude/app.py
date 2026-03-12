# app.py
import os
import sys
import importlib
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# Robust backend import (handles caching & name collisions)
# ------------------------------------------------------------
BACKEND_MODULE_CANDIDATES = ["backend", "rag_backend"]

backend = None
backend_name = None

for name in BACKEND_MODULE_CANDIDATES:
    try:
        backend = importlib.import_module(name)
        backend_name = name
        break
    except Exception:
        continue

if backend is None:
    st.error(
        "❌ Could not import backend. Ensure backend.py (or rag_backend.py) "
        "exists in the same folder as app.py."
    )
    st.stop()

# Validate required backend APIs
required = [
    "ingest_pdf",
    "ingest_images",
    "ingest_csv",
    "ingest_markdown",
    "ingest_audio",
    "ingest_video",
    "answer_query",
]
missing = [r for r in required if not hasattr(backend, r)]
if missing:
    st.error(
        f"❌ Backend module '{backend_name}' is missing: {missing}\n\n"
        f"Loaded from: {getattr(backend, '__file__', 'unknown')}"
    )
    st.stop()

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("🧠 Multimodal RAG Assistant")
st.caption(
    "Upload files → Ingest → Ask query → Get answer + visual evidence (with worklogs)"
)

# Backend info
with st.expander("🧩 Backend Info", expanded=False):
    st.write("Backend module:", backend_name)
    st.write("Backend file:", getattr(backend, "__file__", "unknown"))
    st.write("Python:", sys.version)

# ------------------------------------------------------------
# Worklogs
# ------------------------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
file_type = st.selectbox(
    "Select file type",
    [
        "PDF",
        "Image (PNG/JPG)",
        "Image Folder",
        "CSV",
        "Markdown",
        "Audio",
        "Video",
    ],
)

uploaded = st.file_uploader(
    "Upload file(s) (multiple allowed for images/folders)",
    accept_multiple_files=True,
)

query = st.text_input(
    "Ask a question",
    placeholder="Explain the key points and show the most relevant visual evidence.",
)

k = st.slider("Top-K retrieval", min_value=3, max_value=12, value=6)

col1, col2, col3 = st.columns([2, 1, 1])
btn_ingest = col1.button("📥 Ingest Uploaded Files", type="primary")
btn_ask = col2.button("🚀 Ask Query")
btn_clear = col3.button("🧹 Clear logs")

if btn_clear:
    st.session_state.logs.clear()

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
UPLOAD_DIR = "uploads"
FRAME_DIR = "video_frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ------------------------------------------------------------
# Helper: save uploads
# ------------------------------------------------------------
def save_uploaded_files(uploaded_files):
    saved = []
    for uf in uploaded_files:
        out_path = os.path.join(UPLOAD_DIR, uf.name)
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(out_path)
    return saved

# ------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------
if btn_ingest:
    st.session_state.logs.clear()

    if not uploaded:
        st.error("Please upload at least one file.")
    else:
        paths = save_uploaded_files(uploaded)
        log(f"Saved {len(paths)} file(s) to uploads/")

        try:
            with st.spinner("Ingesting files into vector database..."):
                if file_type == "PDF":
                    log("Ingesting PDF (text + charts)...")
                    backend.ingest_pdf(paths[0])

                elif file_type in ("Image (PNG/JPG)", "Image Folder"):
                    log("Ingesting image(s)...")
                    backend.ingest_images(paths)

                elif file_type == "CSV":
                    log("Ingesting CSV...")
                    backend.ingest_csv(paths[0])

                elif file_type == "Markdown":
                    log("Ingesting Markdown...")
                    backend.ingest_markdown(paths[0])

                elif file_type == "Audio":
                    log("Ingesting audio (Whisper transcription)...")
                    backend.ingest_audio(paths[0])

                elif file_type == "Video":
                    log("Ingesting video (audio + frames)...")
                    backend.ingest_video(paths[0])

                log("✅ Ingestion completed.")
                st.success("✅ Ingestion completed. You can now ask a question.")

        except Exception as e:
            st.error(f"❌ Ingestion failed: {e}")
            log(f"ERROR: {repr(e)}")

# ------------------------------------------------------------
# Query
# ------------------------------------------------------------
if btn_ask:
    if not query.strip():
        st.error("Please enter a query.")
    else:
        st.session_state.logs.clear()
        log("Running query...")
        log(f"Query: {query}")
        log(f"Top-K: {k}")

        try:
            with st.spinner("Retrieving context + invoking Claude..."):
                result = backend.answer_query(query=query, k=k)

            # ---------------- Answer ----------------
            st.subheader("✅ Answer")
            st.write(result["answer"])

            # ---------------- Image Evidence ----------------
            images = result.get("images", [])
            if images:
                st.subheader("🖼️ Relevant Visual Evidence")

                # FIX: decode base64 → bytes → PIL Image
                img_bytes = base64.b64decode(images[0])
                img = Image.open(BytesIO(img_bytes))

                st.image(img, width=700)
            else:
                st.info("No relevant image retrieved for this query.")

            # ---------------- Debug Context ----------------
            with st.expander("🔍 Retrieved Context (Debug)", expanded=False):
                texts = result.get("texts", [])
                st.markdown(f"**Text chunks:** {len(texts)}")
                for i, t in enumerate(texts[:6]):
                    st.markdown(f"**Chunk {i + 1}:**")
                    st.write(t[:1200])

                st.markdown(f"**Images:** {len(images)}")

            log("✅ Query completed successfully.")

        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            log(f"ERROR: {repr(e)}")

# ------------------------------------------------------------
# Logs panel
# ------------------------------------------------------------
with st.expander("📋 Worklogs (what’s happening)", expanded=True):
    if not st.session_state.logs:
        st.write("No logs yet.")
    else:
        for line in st.session_state.logs:
            st.text(line)
