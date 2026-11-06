import pytest
from conftest import wait_ready
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.schemas import ExperimentPlan, TaskRequest
from app.tools import make_tools


def profile_request():
    payload = {"task": "profile", "plan": None, "narrative": ""}
    return {
        **payload,
        "explanation": "Inspect column types, missing values, and candidate targets.",
        "code": f"from ml_runtime import execute\nexecute({payload!r})",
    }


class ScriptedModel(BaseChatModel):
    responses: list

    @property
    def _llm_type(self):
        return "test-only-scripted-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_real_langgraph_interrupt_runs_only_after_approval(client, runner, monkeypatch, decision):
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_ml_task",
                        "args": {"request": profile_request()},
                        "id": "call-profile",
                    }
                ],
            ),
            AIMessage(
                content="The inspection is complete."
                if decision == "approve"
                else "What would you like changed?"
            ),
        ]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    assert client.post("/api/sessions").status_code == 200
    assert (
        client.post(
            "/api/uploads?kind=training", files={"file": ("data.csv", b"x,y\n1,0\n2,1\n")}
        ).status_code
        == 200
    )
    session = next(iter(client.app.state.store.sessions.values()))
    assert client.post("/api/chat", json={"message": "Please inspect my CSV"}).status_code == 200
    wait_ready(session)
    assert session.error is None
    assert session.pending
    assert runner.calls == []
    approval_id = session.pending["id"]
    response = client.post(
        "/api/approvals",
        json={"approval_id": approval_id, "decision": decision, "feedback": "Please explain first"},
    )
    assert response.status_code == 200
    wait_ready(session)
    assert session.error is None
    assert len(runner.calls) == (1 if decision == "approve" else 0)
    assert (
        client.post(
            "/api/approvals", json={"approval_id": approval_id, "decision": "approve"}
        ).status_code
        == 409
    )


def test_generated_code_cannot_escape_supported_runtime():
    request = profile_request()
    for code in [
        "import os\nos.system('id')",
        request["code"] + "\nprint('extra')",
        "from ml_runtime import execute\nexecute({'task': 'report'})",
    ]:
        with pytest.raises(ValueError):
            TaskRequest.model_validate({**request, "code": code})


def test_holdout_cannot_be_used_for_refinement(client, runner):
    client.post("/api/sessions")
    session = next(iter(client.app.state.store.sessions.values()))
    session.dataset, session.profile, session.finalized = "training.csv", {}, True
    payload = {
        "task": "train",
        "plan": ExperimentPlan(target="y", positive_label="1").model_dump(),
        "narrative": "",
    }
    request = TaskRequest(
        **payload,
        explanation="Try a different model after test results",
        code=f"from ml_runtime import execute\nexecute({payload!r})",
    )
    result = make_tools(session, client.app.state.workspace, runner)[0].invoke(
        {"request": request.model_dump()}
    )
    assert result["status"] == "failed"
    assert "finalized" in result["error"]
    assert runner.calls == []
