"""Validate tool proposals before approval or execution, with bounded correction."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError


class InvalidPlanError(ValueError):
    """A planning failure whose message is safe to display to the user."""


class PlanningGuard(AgentMiddleware):
    def __init__(self, tools, max_repairs):
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_repairs = max_repairs

    def proposal_problem(self, response):
        message = next(
            (item for item in reversed(response.result) if isinstance(item, AIMessage)), None
        )
        if message is None:
            return None
        if message.invalid_tool_calls:
            return (
                "Tool arguments must be a valid structured object. Submit one supported tool call."
            )
        if not message.tool_calls:
            return None
        if len(message.tool_calls) != 1:
            return "Propose exactly one action at a time."
        call = message.tool_calls[0]
        tool = self.tool_map.get(call["name"])
        if tool is None:
            return "Choose one supported tool: " + ", ".join(self.tool_map) + "."
        try:
            validated = tool.args_schema.model_validate(call["args"])
        except ValidationError as exc:
            # Never echo model input values or a raw Pydantic exception to the UI/provider.
            details = []
            for error in exc.errors(include_input=False, include_context=False, include_url=False)[
                :4
            ]:
                field = ".".join(map(str, error["loc"])) or "request"
                details.append(f"{field}: {error['msg']}")
            return "Correct the proposed settings: " + "; ".join(details)
        request = getattr(validated, "request", None)
        if call["name"] == "train_model" and request.task != "train":
            return "train_model requires task='train' and a complete training plan."
        if call["name"] == "execute_ml_task" and request.task == "train":
            return "Use train_model for training so the user can approve the plan."
        return None

    def wrap_model_call(self, request, handler):
        for attempt in range(self.max_repairs + 1):
            response = handler(request)
            problem = self.proposal_problem(response)
            if problem is None:
                return response
            if attempt == self.max_repairs:
                raise InvalidPlanError(
                    "I couldn't prepare a valid plan for the proposed action. That action was not run. "
                    "You can continue chatting or ask me to try the plan again."
                )
            # The rejected tool call was never added to graph state or executed.
            # Supply correction guidance without introducing unmatched tool messages.
            request = request.override(
                messages=[
                    *request.messages,
                    HumanMessage(
                        content=(
                            "The previous proposed action was rejected before approval or execution. "
                            + problem
                            + " Do not provide Python code; the backend generates it. Preserve the user's "
                            "requested model, split and objective when correcting the proposal."
                        )
                    ),
                ]
            )
