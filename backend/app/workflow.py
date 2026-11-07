import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import ValidationError

from .agent import build_agent
from .workspace import new_id


def clear_agent_memory(session):
    checkpointer = session.checkpointer
    session.agent = session.checkpointer = None
    session.pending = session.approved_training = None
    if checkpointer:
        try:
            checkpointer.delete_thread(session.id)
        except Exception:  # noqa: BLE001 - cleanup must not strand a failed request
            logging.getLogger(__name__).warning(
                "Agent checkpoint cleanup failed; memory discarded."
            )


def text_content(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))


class Workflow:
    def __init__(self, settings, store, workspace, runner):
        self.settings, self.store, self.workspace, self.runner = settings, store, workspace, runner

    def submit(self, session, message=None, approval=None):
        with session.lock:
            if session.busy:
                raise ValueError("A task is already running")
            if session.closed:
                raise ValueError("Session ended")
            if session.invocations >= 40:
                raise ValueError(
                    "Session interaction limit reached; download outputs and end the session"
                )
            if approval:
                if not session.pending or session.pending["id"] != approval.approval_id:
                    raise ValueError("Approval expired or was already used")
                if approval.decision == "approve":
                    from .schemas import TaskRequest

                    try:
                        session.approved_training = TaskRequest.model_validate(
                            session.pending["actions"][0]["args"]["request"]
                        ).payload()
                    except (ValidationError, KeyError, IndexError, TypeError):
                        session.error = (
                            "This training proposal is no longer valid. "
                            "Ask me to prepare a new plan; no training was started."
                        )
                        session.status = "error"
                        clear_agent_memory(session)
                        raise ValueError(session.error) from None
                decisions = [
                    {
                        "type": approval.decision,
                        **(
                            {"message": approval.feedback or "Please revise the proposal."}
                            if approval.decision == "reject"
                            else {}
                        ),
                    }
                    for _ in session.pending["actions"]
                ]
                payload = Command(resume={"decisions": decisions})
                session.pending = None
                session.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Approved the proposed action."
                            if approval.decision == "approve"
                            else f"Requested revision: {approval.feedback}"
                        ),
                    }
                )
            else:
                if session.pending:
                    raise ValueError(
                        "Approve or request a revision before continuing the conversation"
                    )
                session.failed_runs = 0
                session.approved_training = None
                session.messages.append({"role": "user", "content": message})
                context = {
                    "training_csv_uploaded": bool(session.dataset),
                    "prediction_csv_uploaded": bool(session.prediction_dataset),
                    "profile_completed": session.profile is not None,
                    "model_trained": bool(session.latest_train),
                    "finalized": session.finalized,
                }
                payload = {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Application status (authoritative): {json.dumps(context)}\n"
                                f"User message:\n{message}"
                            )
                        )
                    ]
                }
            session.busy, session.status, session.error = True, "working", None
            session.task_id = new_id()
            session.invocations += 1
        self.store.pool.submit(self._invoke, session, payload)
        return session.task_id

    def _invoke(self, session, payload):
        try:
            if session.agent is None:
                agent, checkpointer = build_agent(
                    self.settings, session, self.workspace, self.runner
                )
                with session.lock:
                    if session.closed:
                        return
                    session.agent, session.checkpointer = agent, checkpointer
            agent = session.agent
            result = agent.invoke(
                payload,
                config={
                    "configurable": {"thread_id": session.id},
                    "recursion_limit": 8 + self.settings.repair_retries * 2,
                },
            )
            with session.lock:
                if session.closed:
                    clear_agent_memory(session)
                    return
                interrupts = result.get("__interrupt__", ())
                if interrupts:
                    value = interrupts[0].value
                    actions = value.get("action_requests", [])
                    if len(actions) != 1:
                        raise ValueError(
                            "The agent must propose exactly one action at a time. Please ask it to retry."
                        )
                    # Validate before showing executable plans, not just after approval.
                    from .schemas import TaskRequest

                    for action in actions:
                        validated = TaskRequest.model_validate(action["args"]["request"])
                        action["args"]["request"] = validated.model_dump()
                    session.pending = {"id": new_id(), "actions": actions}
                    session.status = "approval_required"
                else:
                    last = next(
                        (
                            m
                            for m in reversed(result["messages"])
                            if m.type == "ai" and not getattr(m, "tool_calls", None)
                        ),
                        None,
                    )
                    if last and text_content(last):
                        session.messages.append(
                            {"role": "assistant", "content": text_content(last)}
                        )
                    session.status = "ready"
        except Exception as exc:  # noqa: BLE001 - isolate request failures and redact provider exceptions
            with session.lock:
                if not session.closed:
                    # Avoid putting SDK exception bodies (which may contain secrets) in responses.
                    if isinstance(exc, ValidationError):
                        session.error = (
                            "The proposed plan contained invalid settings. That action was not run. "
                            "Please ask me to revise the plan or try again."
                        )
                    else:
                        session.error = (
                            str(exc)
                            if isinstance(exc, ValueError)
                            else "The agent request failed. Check the model/key and try again."
                        )[:2000]
                    key = self.settings.api_key.get_secret_value()
                    if key:
                        session.error = session.error.replace(key, "[redacted]")
                    session.status = "error"
                    # A failed proposal or interrupted request must not strand the conversation.
                    clear_agent_memory(session)
        finally:
            with session.lock:
                session.busy = False
                if session.closed and session.checkpointer:
                    clear_agent_memory(session)
