from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.hooks import HookProvider, HookRegistry
from strands.hooks import BeforeInvocationEvent, AfterInvocationEvent

model = BedrockModel(
    client_args={"region_name": "us-east-1"},
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    max_tokens=300
)

#logging hook
class LoggingHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.log_start)
        registry.add_callback(AfterInvocationEvent, self.log_end)

    def log_start(self, event: BeforeInvocationEvent) -> None:
        print("\n===== REQUEST START =====")
        print(f"Agent: {event.agent.name}")

    def log_end(self, event: AfterInvocationEvent) -> None:
        print("===== REQUEST END =====\n")

# Create agent
agent = Agent(
    model=model,
    hooks=[LoggingHook()]
)

# Invoke Agent
result = agent("Tell me a 1 line joke about computers.")
print(result.message)
