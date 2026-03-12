from typing import Any, Dict
import json


def pretty_json(data: Dict[str, Any]) -> str:
    """
    Pretty-print a dictionary as formatted JSON.
    """
    return json.dumps(data, indent=2, ensure_ascii=False)


def section(title: str, content: str) -> str:
    """
    Format a response section for CLI / markdown output.
    """
    return f"\n## {title}\n{content.strip()}\n"


def bullet_list(items) -> str:
    """
    Format a list of strings as bullet points.
    """
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)
