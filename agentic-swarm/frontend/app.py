# =========================================================
# PYTHON PATH FIX (REQUIRED FOR STREAMLIT)
# =========================================================
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st

from swarm.swarm_runner import run_finops_swarm
from tools.finops_tools import load_sample_cost_data
from tools.aws_cost_explorer import get_cost_explorer_summary
from frontend.ui_helpers import extract_sections


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Agentic FinOps Dashboard",
    layout="wide",
)

# =========================================================
# HEADER
# =========================================================
st.title("💼 Agentic FinOps Dashboard")
st.caption(
    "Decision-ready cloud cost optimization using autonomous AI agents"
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Analysis Configuration")

data_source = st.sidebar.radio(
    "Cost Data Source",
    ["Sample Data", "AWS Cost Explorer"],
)

run_button = st.sidebar.button("🚀 Run FinOps Analysis")

# =========================================================
# MAIN LOGIC
# =========================================================
if run_button:
    with st.spinner("Agents are analyzing cloud costs..."):
        try:
            # -------------------------------------------------
            # LOAD COST DATA (ENTERPRISE-SAFE FALLBACK)
            # -------------------------------------------------
            try:
                if data_source == "AWS Cost Explorer":
                    cost_data = get_cost_explorer_summary(days=14)
                    source_used = "AWS Cost Explorer"
                else:
                    raise Exception("Force sample data")
            except Exception:
                cost_data = load_sample_cost_data()
                source_used = "Sample JSON"
                st.warning(
                    "AWS Cost Explorer access is not available for this user. "
                    "Falling back to sample cost data."
                )

            # -------------------------------------------------
            # RUN FINOPS SWARM
            # -------------------------------------------------
            result = run_finops_swarm(cost_data)

            # 🔒 Always convert AgentResult → string
            decision_text = str(result.results["decision_agent"].result)

            sections = extract_sections(decision_text)

            # =================================================
            # EXECUTIVE SUMMARY (KPIs)
            # =================================================
            st.subheader("📈 Executive Summary")

            col1, col2, col3 = st.columns(3)

            primary_service = (
                cost_data["services"][0]["service"]
                if cost_data.get("services")
                else "N/A"
            )

            col1.metric("Primary Cost Driver", primary_service)
            col2.metric("Services Analyzed", len(cost_data.get("services", [])))
            col3.metric("Risk Signals", len(cost_data.get("anomalies", [])))

            st.caption(f"Data Source: {source_used}")
            st.divider()

            # =================================================
            # RESULTS TABS
            # =================================================
            tab1, tab2, tab3, tab4 = st.tabs(
                ["🔍 Key Findings", "💡 Optimization Actions", "⚠️ Risks", "🗺 Final Plan"]
            )

            with tab1:
                st.markdown(sections["findings"] or decision_text)

            with tab2:
                st.markdown(sections["actions"] or decision_text)

            with tab3:
                st.markdown(sections["risks"] or decision_text)

            with tab4:
                st.markdown(sections["plan"] or decision_text)

            # =================================================
            # RAW DATA VIEW
            # =================================================
            with st.expander("📂 View Raw Cost Data"):
                st.json(cost_data)

        except Exception as e:
            st.error("❌ FinOps analysis failed")
            st.exception(e)

else:
    st.info("Select a data source and click **Run FinOps Analysis** to begin.")
