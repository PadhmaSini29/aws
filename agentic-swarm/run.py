from swarm.swarm_runner import run_finops_swarm
from tools.aws_cost_explorer import get_cost_explorer_summary
from tools.finops_tools import load_sample_cost_data


def main():
    print("\n🚀 Starting Agentic FinOps Swarm...\n")

    try:
        print("🔍 Fetching real AWS Cost Explorer data...")
        cost_data = get_cost_explorer_summary(days=14)
        source = "AWS Cost Explorer"
    except Exception as e:
        print("⚠️  AWS Cost Explorer not available.")
        print(f"    Reason: {e}")
        print("📦 Falling back to sample cost data...\n")
        cost_data = load_sample_cost_data()
        source = "Sample JSON"

    print(f"✅ Using data source: {source}\n")

    result = run_finops_swarm(cost_data)

    print("\n================ FINAL FINOPS DECISION ================\n")
    print(result.results["decision_agent"].result)
    #print(result.final_output)
    print("\n======================================================\n")


if __name__ == "__main__":
    main()
