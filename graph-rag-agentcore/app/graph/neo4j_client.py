# app/graph/neo4j_client.py

import os
from neo4j import GraphDatabase

_NEO4J_DRIVER = None


def get_driver():
    """
    Returns a singleton Neo4j driver instance.
    """

    global _NEO4J_DRIVER

    if _NEO4J_DRIVER is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        _NEO4J_DRIVER = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=300,
        )

    return _NEO4J_DRIVER
