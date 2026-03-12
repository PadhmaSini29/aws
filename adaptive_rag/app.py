import time
import os
import streamlit as st
import pandas as pd

from config import setup_logging, get_config
from src.agent import create_nl2sql_agent

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive NL2SQL Agent",
    layout="wide"
)

setup_logging()

# ------------------------------------------------------------
# Sidebar: Runtime Settings (ENV OVERRIDES)
# ------------------------------------------------------------
st.sidebar.title("⚙️ Runtime Settings")

st.sidebar.caption("These settings override environment variables for this session.")

engine = st.sidebar.selectbox(
    "Execution Engine",
    options=["postgres", "sqlite"],
    index=0
)

st.sidebar.markdown("### 🧠 Model (Bedrock)")
model_id = st.sidebar.text_input(
    "BEDROCK_MODEL_ID",
    value=os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
)

st.sidebar.markdown("### 🗄️ PostgreSQL Settings")
pg_host = st.sidebar.text_input("PG_HOST", os.environ.get("PG_HOST", "localhost"))
pg_port = st.sidebar.number_input("PG_PORT", value=int(os.environ.get("PG_PORT", 5432)))
pg_database = st.sidebar.text_input("PG_DATABASE", os.environ.get("PG_DATABASE", "wealthmanagement"))
pg_user = st.sidebar.text_input("PG_USER", os.environ.get("PG_USER", "postgres"))
pg_password = st.sidebar.text_input(
    "PG_PASSWORD",
    value=os.environ.get("PG_PASSWORD", ""),
    type="password"
)

st.sidebar.markdown("### 📁 SQLite Settings")
sqlite_path = st.sidebar.text_input(
    "SQLITE_PATH",
    os.environ.get("SQLITE_PATH", r".\data\wealthmanagement.db")
)

apply_settings = st.sidebar.button("✅ Apply settings")

# ------------------------------------------------------------
# Apply sidebar settings → environment variables
# ------------------------------------------------------------
if apply_settings:
    os.environ["BEDROCK_MODEL_ID"] = model_id
    os.environ["PG_HOST"] = pg_host
    os.environ["PG_PORT"] = str(pg_port)
    os.environ["PG_DATABASE"] = pg_database
    os.environ["PG_USER"] = pg_user
    os.environ["PG_PASSWORD"] = pg_password
    os.environ["SQLITE_PATH"] = sqlite_path

    st.sidebar.success("Settings applied. Agent reloaded.")
    st.cache_resource.clear()

# ------------------------------------------------------------
# Load agent (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_agent():
    cfg = get_config()
    agent = create_nl2sql_agent(cfg)
    return cfg, agent

cfg, agent = load_agent()

# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
st.title("🧠 Adaptive Structured RAG — NL2SQL Agent")
st.caption("Natural Language → SQL → Database (PostgreSQL / SQLite)")

question = st.text_area(
    "Ask a question",
    placeholder="How many clients do we have?\nShow all conservative clients\nTotal investment by risk tolerance",
    height=120
)

run_btn = st.button("▶ Run Query", use_container_width=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_dataframe(result):
    cols = result.get("columns") or []
    rows = result.get("rows") or []
    rows = [list(r) if not isinstance(r, list) else r for r in rows]
    if cols and rows:
        return pd.DataFrame(rows, columns=cols)
    return None

# ------------------------------------------------------------
# Run query
# ------------------------------------------------------------
if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL and executing query..."):
            start_time = time.perf_counter()

            try:
                output = agent.answer(
                    question=question.strip(),
                    engine=engine
                )
                end_time = time.perf_counter()

                sql = output["sql"]
                result = output["result"]
                attempts = output["attempts"]

                exec_time_ms = round((end_time - start_time) * 1000, 2)
                row_count = len(result.get("rows", []))

                # ------------------------------------------------------------
                # Results display
                # ------------------------------------------------------------
                st.success("Query executed successfully")

                col1, col2, col3 = st.columns(3)
                col1.metric("⏱ Execution Time (ms)", exec_time_ms)
                col2.metric("📄 Rows Returned", row_count)
                col3.metric("🔁 Attempts", attempts)

                st.markdown("### 🧾 Generated SQL")
                st.code(sql, language="sql")

                df = to_dataframe(result)
                if df is not None:
                    st.markdown("### 📊 Query Results")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No rows returned.")

            except Exception as e:
                st.error(f"Query failed: {e}")
