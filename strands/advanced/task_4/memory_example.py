import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel
from duckduckgo_search import DDGS

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

bedrock_model = BedrockModel(
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    max_tokens=4000,
    temperature=0.5
)

# ------------------------------------------------------
# 2. SIMPLE MEMORY BACKEND (fully local)
# ------------------------------------------------------

class SimpleMemoryBackend:
    """A minimal in-memory + file-backed memory store."""

    def __init__(self, file_path="memory_store.json", max_items=300):
        self.file_path = Path(file_path)
        self.max_items = max_items
        self.memories = self._load()

    def _load(self):
        if not self.file_path.exists():
            return []
        try:
            return json.load(open(self.file_path, "r"))
        except:
            return []

    def _save(self):
        json.dump(self.memories, open(self.file_path, "w"), indent=2)

    def add(self, text: str):
        """Add memory and enforce max size."""
        self.memories.append({"text": text})
        self.memories = self.memories[-self.max_items:]
        self._save()

    def search(self, query: str, top_k=5):
        """Simple relevance search (keyword overlap)."""
        scored = []
        for m in self.memories:
            score = sum(1 for w in query.lower().split() if w in m["text"].lower())
            if score > 0:
                scored.append({"text": m["text"], "score": score})

        # sort by relevance
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def clear(self):
        self.memories = []
        self._save()


# ------------------------------------------------------
# 3. Web Search Helper
# ------------------------------------------------------

def web_search(query: str, max_results=2) -> List[str]:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [r["body"] for r in results]
    except:
        return []


# ------------------------------------------------------
# 4. Format Context for the Agent
# ------------------------------------------------------

def format_context(memories, snippets):
    text = "=== RELEVANT MEMORY ===\n"
    if not memories:
        text += "(none)\n"
    else:
        for m in memories:
            text += f"- {m['text']} (score={m['score']})\n"

    text += "\n=== WEB SEARCH ===\n"
    if not snippets:
        text += "(none)\n"
    else:
        for s in snippets:
            text += f"- {s}\n"

    return text


# ------------------------------------------------------
# 5. Build Agent + Memory Manager
# ------------------------------------------------------

def build_agent():
    agent = Agent(
        system_prompt="You are a helpful assistant with long-term memory.",
        model=bedrock_model,
    )

    memory = SimpleMemoryBackend("memory_store.json")
    agent.memory = memory

    return agent


# ------------------------------------------------------
# 6. MAIN LOOP
# ------------------------------------------------------

def main():
    agent = build_agent()

    print("\n🧠 Long-Term Memory Agent Ready!")
    print("Type something to talk.")
    print("Commands: /clear = reset memory, /exit = quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/clear":
            agent.memory.clear()
            print("[Memory cleared]\n")
            continue

        # 1. Retrieve memory
        memories = agent.memory.search(user_input)

        # 2. Web search
        snippets = web_search(user_input)

        # 3. Build final prompt
        context = format_context(memories, snippets)
        final_prompt = f"{context}\n\nUSER INPUT:\n{user_input}"

        # 4. Get agent response
        result = agent(final_prompt)
        answer = result.message

        print("Assistant:", answer, "\n")

        # 5. Save memory
        agent.memory.add(f"User: {user_input} | Assistant: {answer}")


# ------------------------------------------------------
if __name__ == "__main__":
    main()
