from fastapi import FastAPI
from swarm.swarm_runner import run_finops_swarm

app = FastAPI()

@app.get("/finops/run")
def run():
    result = run_finops_swarm()
    return {
        "status": result.status,
        "final": result.results["decision_agent"].result
    }
