# app/graph/graph_retriever.py

from typing import List

from app.graph.neo4j_client import get_driver


def retrieve_graph_context(question: str, limit: int = 20) -> str:
    """
    Retrieves relationship-based context from Neo4j.
    This is Graph RAG (no embeddings).

    Args:
        question: User question (future: entity extraction)
        limit: Max relationships returned

    Returns:
        String context for agent prompt
    """

    driver = get_driver()

    query = """
    MATCH (a)-[r]->(b)
    RETURN
        labels(a)[0] AS source_type,
        a.name AS source,
        type(r) AS relationship,
        labels(b)[0] AS target_type,
        b.name AS target
    LIMIT $limit
    """

    records: List[str] = []

    with driver.session() as session:
        result = session.run(query, limit=limit)

        for row in result:
            records.append(
                f"{row['source_type']}({row['source']}) "
                f"-[{row['relationship']}]-> "
                f"{row['target_type']}({row['target']})"
            )

    if not records:
        return "No relevant graph relationships found."

    return "\n".join(records)

