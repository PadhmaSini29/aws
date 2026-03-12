from strands import tool
from agents.document_ingestion_agent import build_document_ingestion_agent
from utils.file_loader import load_document
from utils.text_chunker import chunk_text


@tool
def ingest_document(path: str) -> str:
    """
    Ingest a BFSI policy document and extract structured policy information.

    Args:
        path: Local path to a BFSI document (txt, pdf, docx, json)

    Returns:
        STRICT JSON describing document class, sections, rules, and key terms
    """
    # Load raw document
    document = load_document(path)

    # Chunk text to keep prompts lightweight
    chunks = chunk_text(document["text"], max_chars=2000, overlap=200)
    context = "\n\n---\n\n".join(chunks[:3])  # keep cost low

    agent = build_document_ingestion_agent()

    prompt = f"""
Document metadata:
- source_path: {document["source_path"]}
- doc_type: {document["doc_type"]}

Document content:
{context}

Extract structured policy information as instructed.
"""

    response = agent(prompt)
    return str(response)
