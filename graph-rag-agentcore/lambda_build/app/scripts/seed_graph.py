# scripts/seed_graph.py
"""
One-time graph seeding script.

Run (from repo root):
  python scripts/seed_graph.py

Requires:
- Neo4j running (default bolt://localhost:7687)
- Env vars set if not default:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from app.graph.graph_loader import seed_graph


def main() -> None:
    seed_graph()


if __name__ == "__main__":
    main()
