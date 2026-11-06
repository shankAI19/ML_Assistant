from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .llm import build_llm
from .tools import make_tools


def build_agent(settings, session, workspace, runner):
    checkpointer = InMemorySaver()
    prompt = (Path(__file__).parent / "prompts" / "agent.txt").read_text(encoding="utf-8")
    prompt += "\nReport writing instructions:\n" + (
        Path(__file__).parent / "prompts" / "report.txt"
    ).read_text(encoding="utf-8")
    agent = create_agent(
        model=build_llm(settings),
        tools=make_tools(session, workspace, runner, max_repairs=settings.repair_retries),
        system_prompt=prompt,
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "execute_ml_task": {"allowed_decisions": ["approve", "reject"]},
                }
            )
        ],
    )
    return agent, checkpointer
