from typing import Any
from strands import Agent
from strands_tools import calculator
from strands.models.bedrock import BedrockModel
from strands.hooks import HookProvider, HookRegistry
from strands.experimental.hooks import BeforeToolInvocationEvent

model = BedrockModel(
    client_args={"region_name": "us-east-1"},
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    max_tokens=300
)

class ConstantToolArguments(HookProvider):
    def __init__(self, fixed_tool_arguments: dict[str, dict[str, Any]]):
        self._tools_to_fix = fixed_tool_arguments

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeToolInvocationEvent, self._fix_tool_arguments)

    def _fix_tool_arguments(self, event: BeforeToolInvocationEvent):
        if params := self._tools_to_fix.get(event.tool_use["name"]):
            event.tool_use["input"].update(params)

# Force calculator to use precision=1
fix_parameters = ConstantToolArguments({
    "calculator": {"precision": 1}
})

# Create agent
agent = Agent(
    model=model,
    tools=[calculator],
    hooks=[fix_parameters]
)

result = agent("What is 2 / 3?")
print(result.message)
