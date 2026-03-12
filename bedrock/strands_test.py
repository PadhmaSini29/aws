# strands_test.py

from strands import Agent
from strands.models import BedrockModel
from agents.tools import ingest_pdfs, rag_query


def build_agent() -> Agent:
    """
    Create a Strands Agent wired to AWS Bedrock + your custom tools.
    """
    # Bedrock model used by the agent to reason and choose tools
    model = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.3,
        top_p=0.8,
    )

    system_prompt = (
        "You are a helpful PDF RAG assistant. "
        "You have two tools: `ingest_pdfs` to refresh the vector store, "
        "and `rag_query` to answer questions using PGVector + Bedrock. "
        "Always use these tools instead of guessing. "
        "If the answer is not in the documents, say 'I do not know.'"
    )

    agent = Agent(
        model=model,
        tools=[ingest_pdfs, rag_query],
        system_prompt=system_prompt,
    )

    return agent


if __name__ == "__main__":
    agent = build_agent()

    print("\n=== 1) Ingest PDFs into PGVector ===\n")
    resp1 = agent(
        "First, ingest or refresh all PDFs so the knowledge base is up to date. "
        "Then say 'Ingestion complete.'"
    )
    print(resp1)

    print("\n=== 2) Ask a RAG question (Claude) ===\n")
    resp2 = agent(
        "Using your tools, answer this using the documents: "
        "What are the key challenges in EdTech product development?"
    )
    print(resp2)

    print("\n=== 3) Ask another RAG question (Llama explicitly) ===\n")
    # Here we explicitly instruct the agent which tool + model to use
    resp3 = agent(
        "Call the `rag_query` tool with model='llama' to answer: "
        "Why is Scrum a good fit for EdTech projects?"
    )
    print(resp3)
