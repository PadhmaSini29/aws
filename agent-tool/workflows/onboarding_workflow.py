from typing import Dict, Any
import json

from tools.document_ingestion_tool import ingest_document
from tools.policy_tool import interpret_policy
from tools.compliance_tool import validate_compliance
from tools.risk_tool import assess_risk
from tools.decision_tool import decide_action


def run_onboarding_workflow(
    customer_profile: Dict[str, Any],
    policy_document_path: str,
) -> Dict[str, Any]:
    """
    End-to-end BFSI onboarding workflow.

    Steps:
    1) Ingest BFSI policy document
    2) Interpret policy into actionable rules
    3) Validate customer compliance
    4) Assess customer risk
    5) Produce final onboarding decision

    Args:
        customer_profile: Customer data as Python dict
        policy_document_path: Path to BFSI policy (.txt)

    Returns:
        Dict containing outputs from each stage
    """

    # Serialize customer data once
    customer_json = json.dumps(customer_profile, indent=2, ensure_ascii=False)

    # 1️⃣ Document ingestion
    extracted_policy = ingest_document(policy_document_path)

    # 2️⃣ Policy interpretation
    policy_guidance = interpret_policy(extracted_policy)

    # 3️⃣ Compliance validation
    compliance_result = validate_compliance(
        customer_json=customer_json,
        policy_guidance=policy_guidance,
    )

    # 4️⃣ Risk assessment
    risk_result = assess_risk(
        customer_json=customer_json,
        compliance_result_json=compliance_result,
    )

    # 5️⃣ Final decision
    decision_result = decide_action(
        compliance_result_json=compliance_result,
        risk_result_json=risk_result,
    )

    return {
        "policy_extraction": extracted_policy,
        "policy_interpretation": policy_guidance,
        "compliance": compliance_result,
        "risk": risk_result,
        "decision": decision_result,
    }
