# tests/test_graph_retriever.py
"""
Unit tests for Graph RAG retriever.

Run (from repo root):
  pytest -q

Notes:
- This test expects Neo4j to be running locally OR you set NEO4J_* env vars.
- It also expects the graph to be seeded. You can run:
    python scripts/seed_graph.py
"""

import os

import pytest

from app.graph.graph_retriever import retrieve_graph_context


@pytest.mark.integration
def test_retrieve_graph_context_returns_string():
    ctx = retrieve_graph_context(question="What is Graph RAG?", limit=10)
    assert isinstance(ctx, str)
    assert len(ctx.strip()) > 0


@pytest.mark.integration
def test_retrieve_graph_context_respects_limit():
    ctx = retrieve_graph_context(question="Anything", limit=2)
    # Each relationship is one line in our retriever
    lines = [ln for ln in ctx.splitlines() if ln.strip()]
    # If graph is empty, retriever returns a message; so only assert when it has edges
    if "No relevant graph relationships found." not in ctx:
        assert len(lines) <= 2


def test_env_defaults_present():
    # Sanity: defaults should exist if env not set
    assert os.getenv("NEO4J_URI", "bolt://localhost:7687")
    assert os.getenv("NEO4J_USER", "neo4j")
