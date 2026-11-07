from types import SimpleNamespace

import pytest

from app.schemas import ExperimentPlan
from app.tools import make_tools


def upload_csv(client, kind, content=b"x,y\n1,0\n2,1\n", name="data.csv", replace=False):
    url = f"/api/uploads?kind={kind}"
    if replace:
        url += "&replace_training=true"
    return client.post(url, files={"file": (name, content, "text/csv")})


def test_prediction_csv_can_be_uploaded_before_training_or_finalization(client):
    client.post("/api/sessions")
    assert upload_csv(client, "prediction", name="future.csv").status_code == 200
    snapshot = client.get("/api/sessions/current").json()
    assert snapshot["prediction_uploaded"] is True
    assert snapshot["prediction_filename"] == "future.csv"
    assert snapshot["dataset_uploaded"] is False
    assert snapshot["finalized"] is False
    assert upload_csv(client, "training", name="training.csv").status_code == 200
    assert client.get("/api/sessions/current").json()["prediction_uploaded"] is True


def test_prediction_upload_does_not_discard_pending_training_approval(client):
    client.post("/api/sessions")
    upload_csv(client, "training")
    session = next(iter(client.app.state.store.sessions.values()))
    session.pending = {"id": "approval", "actions": []}
    assert upload_csv(client, "prediction").status_code == 200
    assert session.pending["id"] == "approval"


def test_replacement_training_starts_fresh_experiment_without_restarting_session(client, runner):
    client.post("/api/sessions")
    upload_csv(client, "training", name="first.csv")
    upload_csv(client, "prediction")
    session = next(iter(client.app.state.store.sessions.values()))
    old_experiment = session.experiment_id
    session.profile, session.plan = {}, ExperimentPlan(target="y", positive_label="1")
    session.latest_train, session.finalized = "a" * 32, True
    session.runs.append({"id": "old-result"})
    session.pending, session.approved_training = {"id": "approval"}, {"task": "train"}
    session.agent = object()
    deleted = []
    session.checkpointer = SimpleNamespace(delete_thread=deleted.append)
    response = upload_csv(
        client, "training", b"amount,fraud\n5,0\n10,1\n", "second.csv", replace=True
    )
    assert response.status_code == 200
    assert session.experiment_id != old_experiment
    assert session.dataset_filename == "second.csv"
    assert session.profile is session.plan is session.latest_train is None
    assert session.prediction_dataset is session.prediction_filename is None
    assert session.pending is session.approved_training is None
    assert session.agent is session.checkpointer is None
    assert session.finalized is False
    assert session.runs == []
    assert len(session.messages) == 1
    assert deleted == [session.id]
    assert runner.calls == []
    assert client.get("/api/sessions/current").status_code == 200


@pytest.mark.parametrize("kind", ["training", "prediction"])
def test_failed_replacement_preserves_previous_file_and_experiment(client, kind):
    client.post("/api/sessions")
    original = b"x,y\n1,0\n2,1\n"
    assert upload_csv(client, kind, original, "original.csv").status_code == 200
    session = next(iter(client.app.state.store.sessions.values()))
    original_experiment = session.experiment_id
    path = client.app.state.workspace.path(session.id) / "inputs" / f"{kind}.csv"
    assert upload_csv(client, kind, b"", "empty.csv", replace=True).status_code == 400
    assert path.read_bytes() == original
    assert session.experiment_id == original_experiment
    assert session.busy is False
    assert list(path.parent.glob("*.upload")) == []


def test_queued_prediction_data_is_not_executed_before_finalization(client, runner):
    client.post("/api/sessions")
    upload_csv(client, "training")
    upload_csv(client, "prediction")
    session = next(iter(client.app.state.store.sessions.values()))
    session.profile, session.latest_train = {}, "a" * 32
    payload = {"task": "predict", "plan": None, "narrative": ""}
    result = make_tools(session, client.app.state.workspace, runner)[0].invoke(
        {
            "request": {
                **payload,
                "explanation": "Predict on the uploaded file",
                "code": f"from ml_runtime import execute\nexecute({payload!r})",
            }
        }
    )
    assert result["status"] == "failed"
    assert "Finalize" in result["error"]
    assert runner.calls == []
