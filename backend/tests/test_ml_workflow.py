"""Real execution tests require Docker; ML code never runs in the host test process."""

import csv
import io
import json
from zipfile import ZipFile

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
    with ZipFile(io.BytesIO(trained["experiment.zip"])) as bundle:
        assert bundle.read("task.py").decode() == script("train", plan)
        assert bundle.read("data/input.csv") == source
        assert b"pipeline.fit" in bundle.read("ml_tasks/train.py")
        assert b"scikit-learn==" in bundle.read("requirements.txt")
        assert "run.py" in bundle.namelist()
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
        {"task": "profile", "result": json.loads(artifacts["result.json"]), "receipt": receipt},
        {"task": "train", "result": validation, "receipt": receipt},
        {"task": "evaluate", "result": result, "receipt": receipt},
    ]
    report, _ = docker_runner.run(
        sid,
        script(
            "report",
            narrative="This experiment identifies the positive event using the approved features.",
        ),
        {
            **inputs,
            "evidence.json": json.dumps(evidence).encode(),
            "chart_001_train_confusion.json": trained["confusion.json"],
            "chart_001_train_roc.json": trained["roc.json"],
            "chart_001_train_precision_recall.json": trained["precision_recall.json"],
        },
    )
    assert report["report.docx"].startswith(b"PK")
    assert b"Binary classification study report" in report["report.md"]
    assert b"```json" not in report["report.md"]
    assert b"Execution evidence:" not in report["report.md"]
    assert b"Figure 1." in report["report.md"]
    with ZipFile(io.BytesIO(report["report.zip"])) as package:
        assert "report.md" in package.namelist()
        assert "report.docx" in package.namelist()
        assert len([name for name in package.namelist() if name.endswith(".png")]) == 3
    with ZipFile(io.BytesIO(report["report.docx"])) as document:
        assert len([name for name in document.namelist() if name.startswith("word/media/")]) == 3
        xml = document.read("word/document.xml").decode()
        assert 'w:val="Title"' in xml
        assert "Figure 1." in xml
        assert xml.count('w:pStyle w:val="Caption"') == 3
        assert "<w:tbl>" in xml
    for outputs in (artifacts, evaluated, predicted, report):
        assert "experiment.zip" in outputs
    if model == "logistic":
        # Replay the extracted package from a different directory, using its own
        # Python sources and data rather than the installed sandbox entry point.
        replay_code = """import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

root = Path("/tmp/standalone replay")
with ZipFile("/work/experiment.zip") as bundle:
    bundle.extractall(root)
subprocess.run([sys.executable, str(root / "run.py")], cwd="/tmp", check=True)
result = json.loads((root / "data/out/result.json").read_text())
Path("/work/out").mkdir()
Path("/work/out/result.json").write_text(json.dumps(result))
"""
        replayed, _ = docker_runner.run(
            sid, replay_code, {"experiment.zip": trained["experiment.zip"]}
        )
        replay_result = json.loads(replayed["result.json"])
        assert replay_result["validation_metrics"] == validation["validation_metrics"]
    assert (
        docker_runner.client.containers.list(
            all=True, filters={"label": f"ml-assistant.session={sid}"}
        )
        == []
    )


@pytest.mark.docker
@pytest.mark.parametrize("split", ["stratified", "group", "chronological"])
@pytest.mark.parametrize("test_fraction", [0, 0.15])
def test_flexible_splits_and_feature_expansion_inside_docker(docker_runner, split, test_fraction):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["amount", "velocity", "channel", "entity", "time", "fraud"])
    for i in range(1000):
        writer.writerow(
            [
                i % 30,
                i % 11,
                "web" if i % 2 else "store",
                i // 10,
                f"2025-01-01T00:{i // 60:02}:{i % 60:02}Z",
                int(i % 5 == 0),
            ]
        )
    source = buffer.getvalue().encode()
    plan = ExperimentPlan(
        target="fraud",
        positive_label="1",
        split=split,
        split_column={"group": "entity", "chronological": "time"}.get(split),
        excluded_columns=["entity", "time"],
        validation_fraction=0.25,
        test_fraction=test_fraction,
        feature_expansion="interaction_only" if test_fraction else "polynomial",
    ).model_dump()
    sid = "f" * 32
    trained, receipt = docker_runner.run(sid, script("train", plan), {"input.csv": source})
    result = json.loads(trained["result.json"])
    assert result["numeric_features_after_expansion"] == (3 if test_fraction else 5)
    assert (
        result["split_counts"]["test"] == 0
        if not test_fraction
        else result["split_counts"]["test"] > 0
    )
    if split != "group":
        assert abs(result["split_counts"]["validation"] - 250) <= 1
    frozen, _ = docker_runner.run(
        sid, script("evaluate"), {"input.csv": source, "model.joblib": trained["model.joblib"]}
    )
    if not test_fraction:
        assert json.loads(frozen["result.json"])["metrics"] is None
    for evidence in [[], [{"task": "train", "result": result, "receipt": receipt}]]:
        report, _ = docker_runner.run(
            sid,
            script("report", narrative="## Findings\nA **saved** progress report."),
            {"evidence.json": json.dumps(evidence).encode()},
        )
        assert report["report.docx"].startswith(b"PK")
        assert "report.md" in report


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
