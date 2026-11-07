from pathlib import Path

from app.schemas import ExperimentPlan
from app.task_templates.report_content import evidence_markdown, readable_narrative


def study_evidence(test_fraction=0.2):
    plan = ExperimentPlan(
        target="fraud", positive_label="1", test_fraction=test_fraction
    ).model_dump()
    values = {
        "rows": 200,
        "positives": 20,
        "prevalence": 0.1,
        "threshold": 0.5,
        "average_precision": 0.62,
        "precision": 0.6,
        "recall": 0.75,
        "specificity": 170 / 180,
        "f1": 2 / 3,
        "roc_auc": 0.87,
        "accuracy": 0.925,
        "balanced_accuracy": (0.75 + 170 / 180) / 2,
        "alert_rate": 0.125,
        "tn": 170,
        "fp": 10,
        "fn": 5,
        "tp": 15,
    }
    return [
        {
            "task": "profile",
            "result": {
                "rows": 1000,
                "duplicates": 0,
                "columns": [
                    {"name": "amount", "dtype": "float64", "missing": 10, "unique": 500},
                    {
                        "name": "fraud",
                        "dtype": "int64",
                        "missing": 0,
                        "unique": 2,
                        "value_counts": {"0": 900, "1": 100},
                    },
                ],
            },
            "receipt": {},
        },
        {
            "task": "train",
            "result": {
                "model": "logistic",
                "plan": plan,
                "features": ["amount"],
                "split_counts": {"train": 600, "validation": 200, "test": 200},
                "training_metrics": {
                    **values,
                    "rows": 600,
                    "positives": 60,
                    "tn": 510,
                    "fp": 30,
                    "fn": 15,
                    "tp": 45,
                },
                "validation_metrics": values,
                "baseline": {
                    "accuracy_always_negative": 0.9,
                    "recall_always_negative": 0,
                    "constant_score_average_precision": 0.1,
                },
            },
            "receipt": {
                "elapsed_seconds": 1.25,
                "network": "none",
                "sandbox_destroyed": True,
                "code_sha256": "machine-only-digest",
            },
        },
    ]


def test_report_uses_readable_methods_values_and_interpretation_without_json():
    text = evidence_markdown(study_evidence(), "Missed positive cases require analyst review.")
    assert "# Binary classification study report" in text
    assert "| Classifier | Logistic regression |" in text
    assert "| Precision | 60.0% | 60.0% |" in text
    assert "| Recall | 75.0% | 75.0% |" in text
    assert "15 positive cases were correctly flagged and 5 were missed" in text
    assert "10 false alarms" in text
    assert "| Training | 60% | 600 |" in text
    assert "| amount | Numeric | 10 (1.0%) |" in text
    assert "| 1 | 100 | Positive class |" in text
    assert "fitted on training data only" in text
    assert "Missed positive cases require analyst review" in text
    assert "No test performance has been measured" in text
    for machine_text in (
        "```json",
        '"target":',
        '"train":',
        "machine-only-digest",
        "code_sha256",
        "Execution evidence:",
    ):
        assert machine_text not in text


def test_report_distinguishes_finalization_without_test_and_unlabeled_predictions():
    evidence = study_evidence(test_fraction=0)
    evidence += [
        {"task": "evaluate", "result": {"metrics": None}},
        {"task": "predict", "result": {"rows": 50, "alert_rate": 0.2, "metrics": None}},
    ]
    text = evidence_markdown(evidence, "")
    assert "finalized without an internal test assessment" in text
    assert "predictions were generated for 50" in text.lower()
    assert "alert rate was 20.0%" in text
    assert "had no outcome labels" in text
    assert "No test performance has been measured" in text
    assert "has not been finalized" not in text


def test_report_keeps_test_evaluation_separate_and_labels_figures():
    evidence = study_evidence()
    evidence.append(
        {"task": "evaluate", "result": {"metrics": evidence[-1]["result"]["validation_metrics"]}}
    )
    plots = [
        {
            "path": Path("figure_01.png"),
            "title": "Confusion matrix",
            "caption": "Confusion matrix for held-out test data.",
        }
    ]
    text = evidence_markdown(evidence, "", plots)
    assert "latest held-out test assessment" in text
    assert "| Metric | Held-out test |" in text
    assert "No test performance has been measured" not in text
    assert "![Confusion matrix](figure_01.png)" in text
    assert "*Figure 1. Confusion matrix for held-out test data.*" in text


def test_narrative_allows_short_explanatory_python_but_excludes_configuration_dumps():
    narrative = 'The threshold defines a decision rule.\n\n```json\n{"threshold": 0.5}\n```\n\n```python\nflagged = score >= threshold\n```'
    text = readable_narrative(narrative)
    assert "threshold defines a decision rule" in text
    assert "```python" in text
    assert "flagged = score >= threshold" in text
    assert "```json" not in text
    assert '"threshold":' not in text
    assert readable_narrative('{"task": "train"}') == ""
    assert (
        readable_narrative("Observed findings.\n\n{'task': 'train'}\n\nNext steps.")
        == "Observed findings.\n\nNext steps."
    )


def test_requested_split_proportions_are_not_rounded_to_whole_percentages():
    evidence = study_evidence()
    evidence[-1]["result"]["plan"].update(validation_fraction=0.125, test_fraction=0.075)
    text = evidence_markdown(evidence, "")
    assert "| Training | 80% |" in text
    assert "| Validation | 12.5% |" in text
    assert "| Held-out test | 7.5% |" in text
