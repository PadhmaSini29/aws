import os
import boto3
import streamlit as st

from config import PGVECTOR_CONNECTION_STRING, PGVECTOR_COLLECTION_NAME

# Bedrock Embeddings + Models
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock     # Llama 3
from langchain_community.chat_models import BedrockChat  # Claude 3

# PDF loading + splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PGVector Vector Store
from langchain_community.vectorstores.pgvector import PGVector

# Prompts
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# Bedrock Client
# ---------------------------------------------------------
bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)


# ---------------------------------------------------------
# DATA INGESTION (extract text + add metadata)
# ---------------------------------------------------------
def data_ingestion():
    documents = []

    # Ensure data folder exists
    if not os.path.isdir("data"):
        print("⚠️ 'data' folder not found")
        return documents

    for file in os.listdir("data"):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join("data", file)
            loader = PyPDFLoader(pdf_path)

            pdf_docs = loader.load_and_split()
            print(f"📄 Loaded {len(pdf_docs)} pages from {file}")

            # Add metadata including page number
            for idx, doc in enumerate(pdf_docs):
                doc.metadata["page_number"] = idx + 1
                doc.metadata["source"] = f"{file} - Page {idx + 1}"

            documents.extend(pdf_docs)

    print("📘 Total Pages Loaded:", len(documents))

    if not documents:
        return []

    # Split pages into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )

    docs = text_splitter.split_documents(documents)
    print("🧩 Total Chunks Created:", len(docs))

    return docs


# ---------------------------------------------------------
# VECTOR STORE (PGVector)
# ---------------------------------------------------------
def create_pgvector_store(docs):
    """Stores all chunks + embeddings into PostgreSQL PGVector."""
    if not docs:
        print("⚠️ No documents to index.")
        return

    PGVector.from_documents(
        embedding=bedrock_embeddings,
        documents=docs,
        connection_string=PGVECTOR_CONNECTION_STRING,
        collection_name=PGVECTOR_COLLECTION_NAME,
    )


def load_pgvector_store():
    """Loads the existing PGVector collection."""
    return PGVector(
        connection_string=PGVECTOR_CONNECTION_STRING,
        collection_name=PGVECTOR_COLLECTION_NAME,
        embedding_function=bedrock_embeddings,
    )


# ---------------------------------------------------------
# LLM MODELS
# ---------------------------------------------------------
def get_claude_llm():
    """Claude 3 Sonnet (chat model)."""
    return BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 700, "temperature": 0.3},
    )


def get_llama_llm():
    """Llama 3 (instruct model)."""
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 700, "temperature": 0.2},
    )


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

# Claude 3 (Chat format)
claude_prompt_template = """
Human: Use the following context to answer the question in 200–250 words.
If the answer is not found in the context, say "I do not know."

<context>
{context}
</context>

Question: {question}

Assistant:
"""
CLAUDE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=claude_prompt_template,
)

# Llama 3 (Instruction format — cleaned)
llama_prompt_template = """
[INST]
Use the following context to answer the question thoroughly (200–250 words).
If the answer is not in the context, reply "I do not know."

Context:
{context}

Question:
{question}
[/INST]
"""
LLAMA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=llama_prompt_template,
)


# ---------------------------------------------------------
# RAG PIPELINE (clean, de-duplicated, Llama-safe)
# ---------------------------------------------------------
def get_response(llm, vectorstore, query, prompt):
    if not query or not query.strip():
        return "Please enter a question."

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    docs = retriever.invoke(query)

    if not docs:
        return "I do not know. No relevant context was found in the vector store."

    # ---- Remove duplicate pages ----
    unique_docs = []
    seen_pages = set()

    for d in docs:
        page = d.metadata.get("page_number")
        if page not in seen_pages:
            unique_docs.append(d)
            seen_pages.add(page)

    # ---- Build readable context block ----
    context = "\n\n".join(
        f"[Page {d.metadata.get('page_number')}] {d.page_content}"
        for d in unique_docs
    )

    # ---- Format final prompt ----
    final_prompt = prompt.format(
        context=context,
        question=query,
    )

    # ---- Call LLM ----
    raw_output = llm.invoke(final_prompt)

    # Convert AIMessage → text
    if hasattr(raw_output, "content"):
        text = raw_output.content
    else:
        text = str(raw_output)

    # Clean stray tokens from Llama-style outputs
    clean_output = (
        text.replace("[INST]", "")
        .replace("[/INST]", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .strip()
    )

    return clean_output


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
def main():
    st.set_page_config("Chat PDF (Bedrock + PGVector)", layout="wide")
    st.header("📄 Chat with PDF — AWS Bedrock + PGVector RAG System")

    user_question = st.text_input("Ask a question from your PDFs")

    with st.sidebar:
        st.title("⚙️ Vector Database Controls")

        if st.button("📌 Generate / Update Vector Store"):
            with st.spinner("Processing PDFs..."):
                docs = data_ingestion()
                if not docs:
                    st.warning("No PDFs found in the 'data' folder or no text could be extracted.")
                else:
                    create_pgvector_store(docs)
                    st.success("Vector store updated!")

    # Claude
    if st.button("🤖 Answer with Claude 3 Sonnet"):
        with st.spinner("Claude 3 is generating..."):
            try:
                vectorstore = load_pgvector_store()
                llm = get_claude_llm()
                output = get_response(llm, vectorstore, user_question, CLAUDE_PROMPT)
                st.write(output)
            except Exception as e:
                st.error(f"Error while answering with Claude: {e}")

    # Llama
    if st.button("🦙 Answer with Llama 3"):
        with st.spinner("Llama 3 is generating..."):
            try:
                vectorstore = load_pgvector_store()
                llm = get_llama_llm()
                output = get_response(llm, vectorstore, user_question, LLAMA_PROMPT)
                st.write(output)
            except Exception as e:
                st.error(f"Error while answering with Llama: {e}")


if __name__ == "__main__":
    main()
