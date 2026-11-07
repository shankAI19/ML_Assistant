import ast
from types import SimpleNamespace

import pytest
from conftest import wait_ready
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError
from test_approval import ScriptedModel

from app.schemas import TaskRequest
from app.tools import make_tools


def logistic_request(**plan_changes):
    return {
        "task": "train",
        "explanation": "Fit logistic regression with 80/20 training and validation; report precision, recall and F1.",
        "plan": {
            "target": "fraud",
            "positive_label": "1",
            "model": "logistic",
            "validation_fraction": 0.2,
            "test_fraction": 0.0,
            **plan_changes,
        },
    }


def proposed(request, call_id="training"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "train_model", "args": {"request": request}, "id": call_id}],
    )


def prepare_session(client):
    client.post("/api/sessions")
    client.post(
        "/api/uploads?kind=training", files={"file": ("fraud.csv", b"amount,fraud\n10,0\n20,1\n")}
    )
    session = next(iter(client.app.state.store.sessions.values()))
    session.profile = {"rows": 2, "columns": []}
    return session


def test_tool_schema_requires_plan_but_not_python_code(client, runner):
    session = prepare_session(client)
    tools = make_tools(session, client.app.state.workspace, runner)
    schema = convert_to_openai_tool(tools[1])["function"]["parameters"]["properties"]["request"]
    assert "code" not in schema["properties"]
    assert "plan" in schema["properties"]
    request = TaskRequest.model_validate(logistic_request())
    tree = ast.parse(request.code)
    payload = ast.literal_eval(tree.body[1].value.args[0])
    assert payload == request.payload()
    assert payload["plan"]["model"] == "logistic"
    assert payload["plan"]["validation_fraction"] == 0.2
    assert payload["plan"]["test_fraction"] == 0.0
    assert request.model_dump()["code"] == request.code


def test_80_20_logistic_with_malformed_legacy_script_reaches_approval_and_runs(
    client, runner, monkeypatch
):
    request = {**logistic_request(), "code": "```python\nfit_logistic({'test_fraction': 0.2})\n```"}
    model = ScriptedModel(responses=[proposed(request), AIMessage(content="Training completed.")])
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    session = prepare_session(client)
    client.post(
        "/api/chat",
        json={"message": "Fit logistic with 80/20 and give training and validation results."},
    )
    wait_ready(session)
    assert session.error is None
    assert session.pending is not None
    assert runner.calls == []
    approved_script = session.pending["actions"][0]["args"]["request"]["code"]
    assert "fit_logistic" not in approved_script
    assert "'test_fraction': 0.0" in approved_script
    client.post(
        "/api/approvals", json={"approval_id": session.pending["id"], "decision": "approve"}
    )
    wait_ready(session)
    assert session.error is None
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == approved_script
    assert session.plan.model == "logistic"
    assert session.plan.test_fraction == 0.0


def test_invalid_plan_is_corrected_before_any_approval_or_execution(client, runner, monkeypatch):
    model = ScriptedModel(
        responses=[
            proposed(logistic_request(threshold=2), "bad"),
            proposed(logistic_request(), "corrected"),
            AIMessage(content="Complete."),
        ]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    session = prepare_session(client)
    client.post("/api/chat", json={"message": "Fit the logistic baseline."})
    wait_ready(session)
    assert session.error is None
    assert session.pending["actions"][0]["args"]["request"]["plan"]["threshold"] == 0.5
    assert len(model.responses) == 1
    assert runner.calls == []


def test_repeated_invalid_plan_is_contained_and_next_chat_still_works(client, runner, monkeypatch):
    model = ScriptedModel(
        responses=[
            proposed(logistic_request(threshold=2), "bad1"),
            proposed(logistic_request(threshold=2), "bad2"),
            AIMessage(content="Precision describes the correctness of alerts."),
        ]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    session = prepare_session(client)
    client.post("/api/chat", json={"message": "Fit logistic regression."})
    wait_ready(session)
    assert "couldn't prepare a valid plan" in session.error
    assert "input_value" not in session.error
    assert "pydantic" not in session.error.lower()
    assert session.busy is False
    assert session.pending is None
    assert runner.calls == []
    assert len(model.responses) == 1
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/chat", json={"message": "Explain precision."}).status_code == 200
    wait_ready(session)
    assert session.error is None
    assert session.messages[-1]["content"] == "Precision describes the correctness of alerts."
    assert session.dataset == "training.csv"


def test_validation_error_fallback_is_readable_even_if_checkpoint_cleanup_fails(client):
    session = prepare_session(client)

    def invalid(*args, **kwargs):
        TaskRequest.model_validate(logistic_request(threshold=2))

    def failed_cleanup(*args):
        raise RuntimeError("Checkpoint cleanup unavailable")

    session.agent = SimpleNamespace(invoke=invalid)
    session.checkpointer = SimpleNamespace(delete_thread=failed_cleanup)
    client.post("/api/chat", json={"message": "Fit logistic regression."})
    wait_ready(session)
    assert "invalid settings" in session.error
    assert "ValidationError" not in session.error
    assert "input_value" not in session.error
    assert session.agent is session.checkpointer is None
    assert session.busy is False
    assert client.get("/api/health").status_code == 200


def test_invalid_settings_are_still_rejected():
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(logistic_request(validation_fraction=0.9, test_fraction=0.3))


@pytest.mark.parametrize("task", ["profile", "evaluate", "predict", "report"])
def test_invalid_arguments_for_each_non_training_action_are_repaired(
    client, runner, monkeypatch, task
):
    if task == "report":
        name = "generate_report"
        bad, good = {"narrative": {"raw": "object"}}, {"narrative": "Summarize the completed work."}
    else:
        name = "execute_ml_task"
        request = {
            "task": task,
            "explanation": "Perform the requested dataset action.",
            "plan": None,
        }
        good = {"request": request}
        bad = {"request": {**request, "plan": logistic_request()["plan"]}}
    messages = [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
        for args, call_id in ((bad, "invalid"), (good, "corrected"))
    ]
    model = ScriptedModel(
        responses=[*messages, AIMessage(content="The requested action completed.")]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    session = prepare_session(client)
    if task in {"evaluate", "predict"}:
        session.latest_train = "a" * 32
        previous = client.app.state.workspace.path(session.id) / "outputs" / session.latest_train
        previous.mkdir()
        (previous / "model.joblib").write_bytes(b"test-only model input")
    if task == "predict":
        client.post(
            "/api/uploads?kind=prediction", files={"file": ("predict.csv", b"amount\n30\n")}
        )
        session.finalized = True
    client.post("/api/chat", json={"message": "Perform the requested action."})
    wait_ready(session)
    assert session.error is None
    assert session.pending is None
    assert len(runner.calls) == 1
    assert session.runs[-1]["task"] == task
    assert model.responses == []


@pytest.mark.parametrize(
    "malformation", ["invalid_json", "unknown_tool", "multiple_calls", "missing_request"]
)
def test_malformed_tool_proposals_are_repaired_before_any_action_runs(
    client, runner, monkeypatch, malformation
):
    valid = {
        "name": "execute_ml_task",
        "args": {"request": {"task": "profile", "explanation": "Inspect the training dataset."}},
        "id": "corrected",
    }
    if malformation == "invalid_json":
        bad = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "execute_ml_task",
                    "args": "{invalid JSON",
                    "id": "broken",
                    "error": "Invalid JSON",
                }
            ],
        )
    elif malformation == "unknown_tool":
        bad = AIMessage(content="", tool_calls=[{**valid, "name": "unknown_tool", "id": "broken"}])
    elif malformation == "multiple_calls":
        bad = AIMessage(
            content="", tool_calls=[{**valid, "id": "first"}, {**valid, "id": "second"}]
        )
    else:
        bad = AIMessage(content="", tool_calls=[{**valid, "args": {}, "id": "broken"}])
    model = ScriptedModel(
        responses=[
            bad,
            AIMessage(content="", tool_calls=[valid]),
            AIMessage(content="Inspection completed."),
        ]
    )
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    session = prepare_session(client)
    client.post("/api/chat", json={"message": "Inspect the CSV."})
    wait_ready(session)
    assert session.error is None
    assert len(runner.calls) == 1
    assert session.runs[-1]["task"] == "profile"
    assert client.get("/api/health").status_code == 200


def test_invalid_pending_approval_is_cleared_without_leaking_validation_details(
    client, runner, monkeypatch
):
    session = prepare_session(client)
    session.pending = {
        "id": "stale",
        "actions": [{"args": {"request": logistic_request(threshold=2)}}],
    }
    response = client.post("/api/approvals", json={"approval_id": "stale", "decision": "approve"})
    assert response.status_code == 409
    assert "no longer valid" in response.json()["detail"]
    assert "input_value" not in response.text
    assert session.pending is None
    assert session.busy is False
    assert runner.calls == []
    model = ScriptedModel(responses=[AIMessage(content="Ready to prepare a new plan.")])
    monkeypatch.setattr("app.agent.build_llm", lambda _: model)
    assert client.post("/api/chat", json={"message": "Prepare a new plan."}).status_code == 200
    wait_ready(session)
    assert session.error is None


@pytest.mark.parametrize(
    "route,body",
    [
        ("/api/chat", {"message": "Fit logistic."}),
        ("/api/approvals", {"approval_id": "test", "decision": "approve"}),
    ],
)
def test_http_boundary_never_exposes_pydantic_input_dump(client, monkeypatch, route, body):
    prepare_session(client)

    def invalid(*args, **kwargs):
        TaskRequest.model_validate(logistic_request(threshold=2))

    monkeypatch.setattr(client.app.state.workflow, "submit", invalid)
    response = client.post(route, json=body)
    assert response.status_code == 409
    assert "invalid" in response.json()["detail"]
    assert "input_value" not in response.text
    assert "errors.pydantic.dev" not in response.text


def test_task_failure_does_not_return_raw_pydantic_details(client, runner, monkeypatch):
    session = prepare_session(client)

    def invalid(*args, **kwargs):
        TaskRequest.model_validate(logistic_request(threshold=2))

    monkeypatch.setattr(runner, "run", invalid)
    result = make_tools(session, client.app.state.workspace, runner)[0].invoke(
        {"request": {"task": "profile", "explanation": "Inspect the uploaded training data."}}
    )
    assert result["status"] == "failed"
    assert "validation" in result["error"]
    assert "input_value" not in result["error"]
    assert "errors.pydantic.dev" not in result["error"]
    assert client.get("/api/health").status_code == 200
