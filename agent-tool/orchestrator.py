import os
from strands import Agent
from strands_tools import file_write
from config import get_model, ORCHESTRATOR_MODEL_ID
from tools.document_ingestion_tool import ingest_document
from tools.policy_tool import interpret_policy
from tools.compliance_tool import validate_compliance
from tools.risk_tool import assess_risk
from tools.decision_tool import decide_action

ORCHESTRATOR_SYSTEM_PROMPT = """

<role>
You are a BFSI Orchestrator Agent.
Your job is to coordinate specialist agents (tools) to produce
decision-ready BFSI outputs.
You do NOT answer directly unless no tool is required.
</role>

<tools>
Available tools:
- ingest_document(path): Extract structured BFSI policy information
- interpret_policy(policy_text): Convert policy text into business rules
- validate_compliance(customer_json, policy_guidance): Check regulatory compliance
- assess_risk(customer_json, compliance_result_json): Generate risk score
- decide_action(compliance_result_json, risk_result_json): Final decision
- file_write(filename, content): Persist final outputs
</tools>

<routing_rules>
Routing logic:
- If the user asks about what a policy says → ingest_document → interpret_policy
- If the user asks about eligibility or compliance → validate_compliance
- If compliance exists → assess_risk
- If a final action is required → decide_action
- If information is missing → clearly list missing inputs
Always follow BFSI compliance-first principles.
</routing_rules>

<output_expectations>
- Prefer structured, auditable responses
- Final output must be decision-ready
- Avoid conversational or vague language
</output_expectations>
"""

def build_orchestrator() -> Agent:
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    orchestrator = Agent(
        model=get_model(ORCHESTRATOR_MODEL_ID),
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[
            ingest_document,
            interpret_policy,
            validate_compliance,
            assess_risk,
            decide_action,
            file_write,
        ],
    )
    return orchestrator
