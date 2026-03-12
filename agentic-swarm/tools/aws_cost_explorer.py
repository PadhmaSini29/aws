import boto3
from datetime import datetime, timedelta
from typing import Dict, Any


def get_cost_explorer_summary(
    days: int = 14,
    granularity: str = "DAILY"
) -> Dict[str, Any]:
    """
    Fetch AWS cost data grouped by SERVICE from Cost Explorer.

    Requires:
    - AWS credentials configured locally
    - ce:GetCostAndUsage permission
    """

    client = boto3.client("ce")

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    response = client.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity=granularity,
        Metrics=["UnblendedCost"],
        GroupBy=[
            {
                "Type": "DIMENSION",
                "Key": "SERVICE",
            }
        ],
    )

    services = []
    anomalies = []

    for day in response.get("ResultsByTime", []):
        date = day["TimePeriod"]["Start"]
        for group in day.get("Groups", []):
            service_name = group["Keys"][0]
            amount = float(
                group["Metrics"]["UnblendedCost"]["Amount"]
            )

            services.append(
                {
                    "date": date,
                    "service": service_name,
                    "cost": round(amount, 2),
                }
            )

            # Simple anomaly hint
            if amount > 100:
                anomalies.append(
                    {
                        "date": date,
                        "service": service_name,
                        "delta": round(amount, 2),
                        "hint": "High daily spend – validate usage or scaling",
                    }
                )

    return {
        "time_range": f"{start_date} to {end_date}",
        "currency": "USD",
        "services": services,
        "anomalies": anomalies,
    }
