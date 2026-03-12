import json
from strands.multiagent import Swarm

from agents.research_agent import research_agent
from agents.analyst_agent import analyst_agent
from agents.risk_agent import risk_agent
from agents.decision_agent import decision_agent

from tools.finops_tools import top_services_by_cost

def run_finops_swarm(cost_data: dict):
    """
    Runs the Agentic FinOps Swarm on provided cost data
    (real AWS data or sample fallback).
    """

    top3 = top_services_by_cost(cost_data, n=3)

    task = f"""
You are a FinOps Swarm.
Analyze this cost summary and produce decision-ready recommendations.

COST_DATA_JSON:
{json.dumps(cost_data, indent=2)}

TOP_3_SERVICES:
{json.dumps(top3, indent=2)}

Output format:
- Key findings
- Optimization actions (with rough impact)
- Risks & validations
- Final prioritized plan
"""

    swarm = Swarm(
        [
            research_agent,
            analyst_agent,
            risk_agent,
            decision_agent,
        ],
        max_handoffs=8,
        max_iterations=10,
        execution_timeout=240,
        node_timeout=60,
        repetitive_handoff_detection_window=8,
        repetitive_handoff_min_unique_agents=3,
    )

    return swarm(task)


# Optional standalone test
if __name__ == "__main__":
    from tools.finops_tools import load_sample_cost_data

    sample_data = load_sample_cost_data()
    result = run_finops_swarm(sample_data)
    print(result.results["decision_agent"].result)
