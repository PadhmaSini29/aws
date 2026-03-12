import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/support"

st.set_page_config(
    page_title="Agentic Customer Support Engine",
    layout="centered"
)

st.title("🧠 Agentic Customer Support Decision Engine")
st.caption("Multi-Agent Decision System powered by Strands Agent Graph")

query = st.text_area(
    "Customer Query",
    placeholder="Type a customer support message here...",
    height=120
)

if st.button("Analyze"):
    if not query.strip():
        st.warning("Please enter a customer query.")
    else:
        with st.spinner("Agents are collaborating..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    params={"query": query},
                    timeout=30
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection failed: {e}")
                st.stop()

        if response.status_code == 200:
            data = response.json()

            st.success("Decision Generated")

            # ✅ Correct key from backend
            st.subheader("📌 Final Decision")
            st.write(data.get("final_decision", "No decision returned"))

            st.subheader("🔄 Agent Execution Order")
            execution_order = data.get("execution_order", [])
            if execution_order:
                st.write(" → ".join(execution_order))
            else:
                st.write("Execution order not available")

            st.subheader("📊 Token Usage")
            token_usage = data.get("token_usage", {})
            if token_usage:
                st.json(token_usage)
            else:
                st.write("Token usage data not available")

        else:
            st.error(
                f"Backend error occurred "
                f"(Status Code: {response.status_code})"
            )
