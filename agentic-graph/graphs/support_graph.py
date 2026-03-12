from strands.multiagent import GraphBuilder
from agents.intent_agent import intent_agent
from agents.sentiment_agent import sentiment_agent
from agents.knowledge_agent import knowledge_agent
from agents.risk_agent import risk_agent
from agents.decision_agent import decision_agent

def build_support_graph():
    builder = GraphBuilder()

    builder.add_node(intent_agent, "intent")
    builder.add_node(sentiment_agent, "sentiment")
    builder.add_node(knowledge_agent, "knowledge")
    builder.add_node(risk_agent, "risk")
    builder.add_node(decision_agent, "decision")

    builder.add_edge("intent", "sentiment")
    builder.add_edge("intent", "knowledge")
    builder.add_edge("sentiment", "risk")
    builder.add_edge("knowledge", "decision")
    builder.add_edge("risk", "decision")

    builder.set_entry_point("intent")
    return builder.build()
