# agents/tools.py

from strands import tool
from agents.rag_engine import ingest_docs, answer_question


@tool
def ingest_pdfs() -> str:
    """
    Ingest all PDFs from S3 and update the PGVector index.

    Use this tool when:
    - New PDFs are uploaded to S3
    - Existing PDFs are updated
    - The vector database needs rebuilding

    Returns:
        A short status message describing the ingestion result.
    """
    return ingest_docs()


@tool
def rag_query(question: str, model: str = "claude") -> str:
    """
    Answer a question using the RAG pipeline over PDF documents.

    Args:
        question (str): Natural language question about the PDFs.
        model (str): LLM to use. Options:
            - "claude" (default)
            - "llama"

    Returns:
        A detailed answer grounded strictly in the PDF content.
    """
    return answer_question(question, model)
