import json
from pathlib import Path
from typing import Dict, Any

def load_sample_cost_data() -> Dict[str, Any]:
    p = Path(__file__).parent.parent / "data" / "sample_cost.json"
    return json.loads(p.read_text())

def top_services_by_cost(data: Dict[str, Any], n: int = 3):
    services = sorted(data["services"], key=lambda x: x["cost"], reverse=True)
    return services[:n]
