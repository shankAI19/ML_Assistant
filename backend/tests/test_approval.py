import pytest
from conftest import wait_ready
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.schemas import ExperimentPlan, TaskRequest
from app.tools import make_tools


def profile_request(task="profile"):
    payload = {
        "task": task,
        "plan": ExperimentPlan(target="y", positive_label="1").model_dump()
        if task == "train"
        else None,
        "narrative": "",
    }
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
                        "name": "train_model",
                        "args": {"request": profile_request("train")},
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
    session.profile = {}
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
    session.approved_training = request.payload()
    result = make_tools(session, client.app.state.workspace, runner)[1].invoke(
        {"request": request.model_dump()}
    )
    assert result["status"] == "failed"
    assert "finalized" in result["error"]
    assert runner.calls == []


def test_inspection_does_not_interrupt(client, runner, monkeypatch):
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_ml_task",
                        "args": {"request": profile_request()},
                        "id": "profile",
                    }
                ],
            ),
            AIMessage(content="Inspection complete."),
        ]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    client.post("/api/sessions")
    client.post("/api/uploads?kind=training", files={"file": ("data.csv", b"x,y\n1,0\n2,1\n")})
    session = next(iter(client.app.state.store.sessions.values()))
    client.post("/api/chat", json={"message": "Inspect this CSV"})
    wait_ready(session)
    assert session.error is None
    assert session.pending is None
    assert len(runner.calls) == 1


def test_training_cannot_bypass_approval(client, runner):
    client.post("/api/sessions")
    session = next(iter(client.app.state.store.sessions.values()))
    tools = make_tools(session, client.app.state.workspace, runner)
    for index in (0, 1):
        result = tools[index].invoke({"request": profile_request("train")})
        assert result["status"] == "failed"
    assert runner.calls == []


def test_report_without_dataset_or_model(client, runner):
    client.post("/api/sessions")
    session = next(iter(client.app.state.store.sessions.values()))
    result = make_tools(session, client.app.state.workspace, runner)[2].invoke(
        {"narrative": "## Progress\nNo training yet."}
    )
    assert result["status"] == "completed"
    assert session.pending is None
    assert set(runner.calls[0][2]) == {"evidence.json"}


@pytest.mark.parametrize("changed_plan", [False, True])
def test_failed_round_reuses_approval_only_for_identical_plan(
    client, runner, monkeypatch, changed_plan
):
    first = profile_request("train")
    second = profile_request("train")
    if changed_plan:
        second["plan"]["threshold"] = 0.7
        payload = {key: second[key] for key in ("task", "plan", "narrative")}
        second["code"] = f"from ml_runtime import execute\nexecute({payload!r})"

    def call(request, id):
        return AIMessage(
            content="", tool_calls=[{"name": "train_model", "args": {"request": request}, "id": id}]
        )

    model = ScriptedModel(
        responses=[call(first, "first"), call(second, "second"), AIMessage(content="Complete")]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    client.post("/api/sessions")
    client.post("/api/uploads?kind=training", files={"file": ("data.csv", b"x,y\n1,0\n2,1\n")})
    session = next(iter(client.app.state.store.sessions.values()))
    session.profile = {}
    original = runner.run
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Transient sandbox failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run", fail_once)
    client.post("/api/chat", json={"message": "Train the agreed plan"})
    wait_ready(session)
    approval_id = session.pending["id"]
    client.post("/api/approvals", json={"approval_id": approval_id, "decision": "approve"})
    wait_ready(session)
    assert session.error is None
    assert len(attempts) == (1 if changed_plan else 2)
    assert bool(session.pending) == changed_plan
