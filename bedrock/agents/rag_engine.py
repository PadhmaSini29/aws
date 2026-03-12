import os
import boto3

from config import (
    PGVECTOR_CONNECTION_STRING,
    PGVECTOR_COLLECTION_NAME,
)

# --- S3 loader ---
from agents.s3_loader import download_pdfs

# --- LangChain / Bedrock ---
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import PromptTemplate


# =====================================================
# Bedrock Client + Embeddings
# =====================================================
bedrock = boto3.client("bedrock-runtime")

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)


# =====================================================
# INGEST DOCUMENTS (FROM S3 → PGVECTOR)
# =====================================================
def ingest_docs():
    """
    Downloads PDFs from S3, splits them, and stores embeddings in PGVector.
    """

    pdf_files = download_pdfs(local_dir="data")
    documents = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for idx, doc in enumerate(pages):
            doc.metadata["page_number"] = idx + 1
            doc.metadata["source"] = f"{os.path.basename(pdf_path)} - Page {idx+1}"

        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=PGVECTOR_CONNECTION_STRING,
        collection_name=PGVECTOR_COLLECTION_NAME
    )

    return f"Ingested {len(chunks)} chunks from S3 PDFs."


# =====================================================
# LOAD VECTOR STORE
# =====================================================
def load_store():
    return PGVector(
        connection_string=PGVECTOR_CONNECTION_STRING,
        collection_name=PGVECTOR_COLLECTION_NAME,
        embedding_function=embeddings
    )


# =====================================================
# LLM MODELS
# =====================================================
def claude_llm():
    return BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={"temperature": 0.3, "max_tokens": 600}
    )


def llama_llm():
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"temperature": 0.2, "max_gen_len": 600}
    )


# =====================================================
# PROMPTS
# =====================================================
CLAUDE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Human: Use the following context to answer the question in 200–250 words.
If the answer is not found in the context, say "I do not know."

<context>
{context}
</context>

Question: {question}

Assistant:
"""
)

LLAMA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
[INST]
Use the following context to answer the question in 200–250 words.
If the answer is not in the context, reply "I do not know."

Context:
{context}

Question:
{question}
[/INST]
"""
)


# =====================================================
# RAG QUERY
# =====================================================
def answer_question(query: str, model: str = "claude"):
    """
    Answers a question using PGVector + Bedrock LLM.
    model = "claude" | "llama"
    """

    store = load_store()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    docs = retriever.invoke(query)

    # De-duplicate pages
    seen_pages = set()
    unique_docs = []

    for d in docs:
        page = d.metadata.get("page_number")
        if page not in seen_pages:
            unique_docs.append(d)
            seen_pages.add(page)

    context = "\n\n".join(
        f"[Page {d.metadata.get('page_number')}] {d.page_content}"
        for d in unique_docs
    )

    prompt = CLAUDE_PROMPT if model == "claude" else LLAMA_PROMPT
    llm = claude_llm() if model == "claude" else llama_llm()

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    result = llm.invoke(final_prompt)

    # Normalize output
    return result.content if hasattr(result, "content") else str(result)
