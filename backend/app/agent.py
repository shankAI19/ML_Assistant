from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .llm import build_llm
from .planning import PlanningGuard
from .schemas import TaskRequest
from .tools import make_tools


def build_agent(settings, session, workspace, runner):
    def needs_training_approval(call):
        request = TaskRequest.model_validate(call.tool_call["args"]["request"])
        # Only an identical failed round can reuse its approval, within the same turn.
        return not (session.failed_runs and request.payload() == session.approved_training)

    checkpointer = InMemorySaver()
    prompt = (Path(__file__).parent / "prompts" / "agent.txt").read_text(encoding="utf-8")
    prompt += "\nReport writing instructions:\n" + (
        Path(__file__).parent / "prompts" / "report.txt"
    ).read_text(encoding="utf-8")
    tools = make_tools(session, workspace, runner, max_repairs=settings.repair_retries)
    agent = create_agent(
        model=build_llm(settings),
        tools=tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
        middleware=[
            PlanningGuard(tools, max_repairs=settings.repair_retries),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "train_model": {
                        "allowed_decisions": ["approve", "reject"],
                        "when": needs_training_approval,
                    },
                }
            ),
        ],
    )
    return agent, checkpointer
