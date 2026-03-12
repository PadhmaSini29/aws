import streamlit as st
import json
import tempfile
from pathlib import Path

from orchestrator import build_orchestrator


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BFSI Agentic Knowledge System",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 BFSI Agentic Knowledge System")
st.caption(
    "Agentic AI using Strands Agents + AWS Bedrock | "
    "From Policies → Compliance → Risk → Decisions"
)

st.divider()

# -----------------------------
# Initialize Orchestrator
# -----------------------------
@st.cache_resource
def load_orchestrator():
    return build_orchestrator()


orchestrator = load_orchestrator()

# -----------------------------
# File Upload Section
# -----------------------------
st.subheader("📄 Upload BFSI Inputs")

policy_file = st.file_uploader(
    "Upload BFSI Policy (TXT)",
    type=["txt"],
    help="RBI circulars, loan policies, internal SOPs"
)

customer_file = st.file_uploader(
    "Upload Customer Profile (JSON)",
    type=["json"],
    help="Customer KYC, income, credit profile"
)

st.divider()

# -----------------------------
# Question Input
# -----------------------------
st.subheader("🧠 Ask a Question")

query = st.text_area(
    "Example queries:",
    height=120,
    placeholder=(
        "• Explain the uploaded policy\n"
        "• Is this customer compliant with the policy?\n"
        "• Can this customer be approved? Give final decision\n"
    ),
)

# -----------------------------
# Helper: Save uploaded files
# -----------------------------
def save_uploaded_file(uploaded_file, suffix: str) -> str:
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / f"uploaded{suffix}"
    file_path.write_bytes(uploaded_file.read())
    return str(file_path)

# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run Agentic Workflow", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Agents are thinking..."):
            try:
                # Save uploaded files (if any)
                policy_path = None
                customer_path = None

                if policy_file:
                    policy_path = save_uploaded_file(policy_file, ".txt")

                if customer_file:
                    customer_path = save_uploaded_file(customer_file, ".json")

                # Auto-augment query with file paths
                final_query = query

                if policy_path and "policy" not in query.lower():
                    final_query += f"\n\nPolicy file path: {policy_path}"

                if customer_path and "customer" not in query.lower():
                    final_query += f"\n\nCustomer file path: {customer_path}"

                # Call orchestrator
                response = orchestrator(final_query)

                st.success("Workflow completed successfully")

                st.subheader("📊 Agentic Output")
                st.code(response, language="json")

            except Exception as e:
                st.error(f"Error during execution: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Built with ❤️ using Strands Agents SDK | AWS Bedrock | "
    "Enterprise Agentic AI Architecture"
)
