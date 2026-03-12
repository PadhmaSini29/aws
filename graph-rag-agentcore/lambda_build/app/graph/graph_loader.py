# app/graph/graph_loader.py

from app.graph.neo4j_client import get_driver


def seed_graph() -> None:
    """
    Seed the Neo4j database with sample Graph-RAG knowledge.
    Run once or during development.
    """

    driver = get_driver()

    query = """
    MERGE (rag:Concept {name: 'RAG'})
    MERGE (graph_rag:Concept {name: 'Graph RAG'})
    MERGE (neo4j:Service {name: 'Neo4j'})
    MERGE (bedrock:Service {name: 'Amazon Bedrock'})
    MERGE (agentcore:Service {name: 'AgentCore'})
    MERGE (assistant:UseCase {name: 'Knowledge Assistant'})

    MERGE (graph_rag)-[:EXTENDS]->(rag)
    MERGE (neo4j)-[:SUPPORTS]->(graph_rag)
    MERGE (bedrock)-[:USED_FOR]->(assistant)
    MERGE (agentcore)-[:RUNS]->(assistant)
    """

    with driver.session() as session:
        session.run(query)

    print("✅ Graph data seeded successfully")


if __name__ == "__main__":
    seed_graph()

