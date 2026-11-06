"""Real execution tests require Docker; ML code never runs in the host test process."""

import csv
import io
import json

import pytest

from app.config import Settings
from app.sandbox import SandboxRunner
from app.schemas import ExperimentPlan


@pytest.fixture
def docker_runner():
    runner = SandboxRunner(Settings())
    if not runner.available():
        pytest.skip("Docker daemon and built ml-assistant-sandbox image are required")
    return runner


def script(task, plan=None, narrative=""):
    payload = {"task": task, "plan": plan, "narrative": narrative}
    return f"from ml_runtime import execute\nexecute({payload!r})"


@pytest.mark.docker
@pytest.mark.parametrize("model", ["logistic", "xgboost", "lightgbm", "random_forest"])
def test_train_test_predict_and_report_inside_docker(docker_runner, model):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["amount", "channel", "fraud"])
    for i in range(800):
        positive = i % 7 == 0
        writer.writerow(
            [
                "" if i % 41 == 0 else (100 + i % 50 if positive else i % 50),
                "web" if i % 2 else "store",
                int(positive),
            ]
        )
    source = buffer.getvalue().encode()
    sid = "d" * 32
    artifacts, receipt = docker_runner.run(sid, script("profile"), {"input.csv": source})
    plan = ExperimentPlan(target="fraud", positive_label="1", model=model, trees=30).model_dump()
    trained, _ = docker_runner.run(sid, script("train", plan), {"input.csv": source})
    validation = json.loads(trained["result.json"])
    assert validation["evaluation_partition"] == "validation"
    assert validation["validation_metrics"]["rows"] > 0
    inputs = {"input.csv": source, "model.joblib": trained["model.joblib"]}
    evaluated, _ = docker_runner.run(sid, script("evaluate"), inputs)
    result = json.loads(evaluated["result.json"])
    assert result["evaluation_partition"] == "held-out test"
    assert 0 <= result["metrics"]["average_precision"] <= 1
    predicted, _ = docker_runner.run(
        sid,
        script("predict"),
        {**inputs, "prediction.csv": b"amount,channel\n125,new-channel\n20,web\n"},
    )
    assert json.loads(predicted["result.json"])["metrics"] is None
    assert b"positive_class_score" in predicted["predictions.csv"]
    evidence = [
        {"task": "profile", "result": json.loads(artifacts["result.json"]), "receipt": receipt}
    ]
    report, _ = docker_runner.run(
        sid,
        script(
            "report",
            narrative="This experiment identifies the positive event using the approved features.",
        ),
        {**inputs, "evidence.json": json.dumps(evidence).encode()},
    )
    assert report["report.docx"].startswith(b"PK")
    assert (
        docker_runner.client.containers.list(
            all=True, filters={"label": f"ml-assistant.session={sid}"}
        )
        == []
    )


@pytest.mark.docker
def test_infinite_loop_is_killed_and_container_removed(docker_runner):
    sid = "e" * 32
    with pytest.raises(RuntimeError, match="time limit"):
        docker_runner.run(sid, "while True:\n    pass\n", {}, timeout=2)
    assert (
        docker_runner.client.containers.list(
            all=True, filters={"label": f"ml-assistant.session={sid}"}
        )
        == []
    )
