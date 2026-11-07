import json

import pytest

from app.schemas import ExperimentPlan, TaskRequest
from app.task_templates import report


def test_split_and_feature_options_validate():
    plan = ExperimentPlan(
        target="y",
        positive_label="1",
        validation_fraction=0.25,
        test_fraction=0,
        feature_expansion="interaction_only",
    )
    assert plan.test_fraction == 0
    for options in [
        {"validation_fraction": 0.8, "test_fraction": 0.3},
        {"validation_fraction": 0},
        {"model": "xgboost", "feature_expansion": "polynomial"},
        {"polynomial_degree": 4},
    ]:
        with pytest.raises(ValueError):
            ExperimentPlan(target="y", positive_label="1", **options)


def test_code_accepts_equivalent_plan_defaults_but_rejects_changed_threshold():
    payload = {"task": "train", "plan": {"target": "y", "positive_label": "1"}, "narrative": ""}
    code = f"from ml_runtime import execute\nexecute({payload!r})"
    request = TaskRequest(**payload, code=code, explanation="Train the agreed logistic baseline")
    assert request.plan.test_fraction == 0.2
    with pytest.raises(ValueError):
        TaskRequest(
            **{**payload, "plan": {**payload["plan"], "threshold": 0.8}},
            code=code,
            explanation="Train the agreed logistic baseline",
        )


@pytest.mark.parametrize("stage", ["empty", "profile", "train", "evaluate", "predict"])
def test_report_progress_and_markdown_survive_conversion_failure(tmp_path, monkeypatch, stage):
    evidence = []
    if stage != "empty":
        evidence.append(
            {
                "task": stage,
                "result": {
                    "rows": 100,
                    "columns": [],
                    "validation_metrics": {"recall": 0.75} if stage == "train" else None,
                    "note": "Observed evidence",
                },
                "receipt": {"code_sha256": "test"},
            }
        )
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(report, "OUT", tmp_path)
    original_read = report.Path.read_text
    monkeypatch.setattr(
        report.Path,
        "read_text",
        lambda path, **kw: original_read(
            source if path == report.Path("/work/evidence.json") else path, **kw
        ),
    )
    monkeypatch.setattr(report, "saved_plots", list)

    def fail(*args):
        raise RuntimeError("Conversion unavailable")

    monkeypatch.setattr(report, "to_docx", fail)
    result = report.run({"narrative": "## Findings\nA **progress** report."})
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Findings" in text
    assert result["report"] is None
    assert result["warnings"]
    if stage == "train":
        assert "0.7500" in text
        assert "not been finalized" in text
    if stage == "empty":
        assert "No model has been trained" in text
