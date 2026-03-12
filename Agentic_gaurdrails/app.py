import streamlit as st
from langchain_core.messages import AIMessage

# Import your backend agent runner
from langgraph_guardrails import run_agent


# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Agentic AI with Bedrock Guardrails",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Agentic AI with Amazon Bedrock Guardrails")
st.caption(
    "LangGraph-powered AI agent with enforced financial & PII safety using Amazon Bedrock Guardrails"
)

st.divider()


# ------------------------------------
# User Input Section
# ------------------------------------
query = st.text_area(
    label="Enter your question",
    placeholder="Example: What is the current status of my loan? My account ID is ACC612330",
    height=120
)

run_button = st.button("🚀 Run Agent", type="primary")


# ------------------------------------
# Run Agent
# ------------------------------------
if run_button:
    if not query.strip():
        st.warning("Please enter a query before running the agent.")
    else:
        with st.spinner("Running agent with guardrails enforced..."):
            try:
                result = run_agent(query)

                # Extract final AI message safely
                final_answer = None
                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content
                        break

                if final_answer:
                    st.success("Agent completed successfully")

                    st.subheader("🤖 Agent Response")
                    st.write(final_answer)
                else:
                    st.warning("No AI response was generated.")

            except Exception as e:
                st.error("An unexpected error occurred while running the agent.")
                st.exception(e)


st.divider()


# ------------------------------------
# Footer
# ------------------------------------
st.caption(
    "Built with LangGraph • Amazon Bedrock • Guardrails • Streamlit"
)
