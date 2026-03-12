import os
import streamlit as st
from dotenv import load_dotenv

from scripts.utils import get_opensearch_client, generate_embedding
from scripts.rag import rag_answer

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "books")
client = get_opensearch_client()

st.set_page_config(page_title="📚 Book Search + RAG", layout="centered")
st.title("📚 Book Semantic Search + RAG")
st.caption("OpenSearch (keyword/vector/hybrid) + Bedrock (Titan embeddings + Claude)")

tab_search, tab_rag = st.tabs(["🔎 Search", "💬 Ask (RAG)"])

# ---------------- SEARCH TAB ----------------
with tab_search:
    query = st.text_input("Search books", placeholder="e.g. habits and discipline, surveillance freedom")
    search_type = st.radio("Search type", ["Keyword", "Semantic", "Hybrid"], horizontal=True)
    top_k = st.slider("Top K results", 1, 8, 3)

    def keyword_search(q):
        return client.search(
            index=INDEX_NAME,
            body={"size": top_k, "query": {"multi_match": {"query": q, "fields": ["title^2", "author", "description"]}}}
        )

    def semantic_search(q):
        vec = generate_embedding(q)
        return client.search(
            index=INDEX_NAME,
            body={"size": top_k, "query": {"knn": {"embedding": {"vector": vec, "k": top_k}}}}
        )

    def hybrid_search(q):
        vec = generate_embedding(q)
        return client.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {"multi_match": {"query": q, "fields": ["title^2", "author", "description"]}},
                            {"knn": {"embedding": {"vector": vec, "k": top_k}}},
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
        )

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            if search_type == "Keyword":
                response = keyword_search(query)
            elif search_type == "Semantic":
                response = semantic_search(query)
            else:
                response = hybrid_search(query)

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            st.warning("No results found.")
        else:
            st.subheader("Results")
            for h in hits:
                book = h["_source"]
                st.markdown(f"### 📖 {book.get('title','')}")
                st.markdown(f"**Author:** {book.get('author','')}")
                st.markdown(book.get("description", ""))
                st.caption(f"Score: {h.get('_score')}")
                st.divider()

# ---------------- RAG TAB ----------------
with tab_rag:
    question = st.text_area("Ask a question (RAG)", placeholder="e.g. Suggest a book to improve focus and explain why.")
    k = st.slider("How many books to retrieve for context (citations)", 2, 8, 4)

    if st.button("Get Answer") and question:
        with st.spinner("Retrieving + generating answer..."):
            result = rag_answer(question, k=k)

        st.subheader("Answer")
        st.write(result["answer"] if result["answer"] else "No answer returned.")

        st.subheader("Citations (retrieved sources)")
        if not result["sources"]:
            st.info("No sources retrieved.")
        else:
            for i, s in enumerate(result["sources"], start=1):
                st.markdown(f"**[{i}] {s['title']}** — {s['author']}")
                st.caption(s.get("description", ""))
                st.caption(f"id={s.get('id')} | score={s.get('score')}")
                st.divider()
