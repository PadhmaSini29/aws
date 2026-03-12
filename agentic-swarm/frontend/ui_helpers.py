def extract_sections(text):
    """
    Safely extract sections from agent output.
    Accepts AgentResult or string.
    """

    # 🔒 SAFETY: always convert to string
    text = str(text)

    sections = {
        "findings": "",
        "actions": "",
        "risks": "",
        "plan": "",
    }

    current = None

    for line in text.splitlines():
        l = line.lower()

        if "key finding" in l:
            current = "findings"
            continue
        elif "optimization" in l:
            current = "actions"
            continue
        elif "risk" in l:
            current = "risks"
            continue
        elif "final" in l or "plan" in l:
            current = "plan"
            continue

        if current:
            sections[current] += line + "\n"

    return sections
