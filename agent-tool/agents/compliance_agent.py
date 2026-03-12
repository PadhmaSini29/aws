from strands import Agent
from config import get_model, TOOL_MODEL_ID

COMPLIANCE_PROMPT = """
You are a BFSI compliance validation specialist.

You will be given:
- customer profile (JSON)
- policy guidance / rules (text or JSON)

Task:
Validate whether the customer profile satisfies policy requirements.
Be conservative: if required data is missing, mark NEEDS_REVIEW and list missing info.

Output format:
Return STRICT JSON only:
{
  "status": "COMPLIANT|NON_COMPLIANT|NEEDS_REVIEW",
  "violations": [
    {
      "rule": "What rule is violated",
      "evidence": "What in the customer data or policy caused this"
    }
  ],
  "missing_info": ["..."],
  "severity": "LOW|MEDIUM|HIGH",
  "notes": "Short explanation"
}
"""

def build_compliance_agent() -> Agent:
    return Agent(
        model=get_model(TOOL_MODEL_ID),
        system_prompt=COMPLIANCE_PROMPT,
    )
