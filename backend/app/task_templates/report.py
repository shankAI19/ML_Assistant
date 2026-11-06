import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .common import OUT, metrics
from .evaluate import evaluate_bundle


def run(payload):
    bundle, y, scores = evaluate_bundle()
    plan = bundle["plan"]
    observed = metrics(y, scores, plan["threshold"])
    evidence = json.loads(Path("/work/evidence.json").read_text(encoding="utf-8"))
    document = Document()
    section = document.sections[0]
    section.header.paragraphs[0].text = "ML ASSISTANT  /  FINAL EXPERIMENT REPORT"
    document.add_heading("Your classification report", 0)
    document.add_paragraph(f"Target: {plan['target']}  |  Positive event: {plan['positive_label']}")
    document.add_heading("What we learned", level=1)
    for paragraph in payload["narrative"].split("\n\n"):
        if paragraph.strip():
            document.add_paragraph(paragraph.strip())
    document.add_heading("Dataset and preparation", level=1)
    profile = next((e["result"] for e in evidence if e["task"] == "profile"), {})
    document.add_paragraph(
        f"Uploaded rows: {profile.get('rows', 'unknown')}. Features used: {len(bundle['features'])}. "
        "Missing numeric values were imputed from training medians; categorical values were imputed and one-hot encoded. "
        "Numeric scaling was applied for logistic regression. All preprocessing was fitted on training data only."
    )
    document.add_paragraph(
        f"Model: {plan['model']}. Split: {plan['split']}. Seed: {plan['seed']}. "
        f"Positive-class weighting: {plan['class_weighted']}. Decision threshold: {plan['threshold']}."
    )
    document.add_paragraph("Features: " + ", ".join(bundle["features"]))
    document.add_paragraph(
        "Excluded columns: " + (", ".join(plan["excluded_columns"]) or "None explicitly selected")
    )
    document.add_heading("Computed performance", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Light Shading Accent 1"
    for cell, title in zip(
        table.rows[0].cells, ["Metric", "Development validation", "Held-out test"]
    ):
        cell.text = title
    for key in [
        "rows",
        "positives",
        "prevalence",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "alert_rate",
        "false_positives_per_1000",
    ]:
        row = table.add_row().cells
        row[0].text = key.replace("_", " ").title()
        for cell, value in zip(row[1:], [bundle["validation_metrics"].get(key), observed.get(key)]):
            cell.text = (
                "Unavailable"
                if value is None
                else f"{value:.4f}"
                if isinstance(value, float)
                else str(value)
            )
    document.add_paragraph(
        "Average precision is calculated using sklearn.metrics.average_precision_score; it is not trapezoidal PR area. "
        "Validation informed model selection. Test results were obtained after the recipe was frozen. "
        "The evaluated model was retained without a full-data refit."
    )
    sns.set_theme(style="whitegrid")
    for name, display in [("roc", RocCurveDisplay), ("precision_recall", PrecisionRecallDisplay)]:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        display.from_predictions(y, scores, ax=axis)
        figure.tight_layout()
        path = OUT / f"{name}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        document.add_picture(str(path), width=Inches(5.8))
    figure, axis = plt.subplots(figsize=(5, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        scores >= plan["threshold"],
        ax=axis,
        colorbar=False,
        display_labels=["Negative", "Positive"],
    )
    figure.tight_layout()
    figure.savefig(OUT / "confusion.png", dpi=140)
    plt.close(figure)
    document.add_picture(str(OUT / "confusion.png"), width=Inches(5.3))
    predictions = [e["result"] for e in evidence if e["task"] == "predict"]
    if predictions:
        document.add_heading("Uploaded prediction data", level=1)
        for prediction in predictions:
            document.add_paragraph(
                f"Rows: {prediction['rows']}; alert rate: {prediction['alert_rate']:.4f}. {prediction['note']}"
            )
            if prediction.get("metrics"):
                document.add_paragraph(json.dumps(prediction["metrics"], indent=2))
    document.add_heading("Limitations and next steps", level=1)
    for warning in bundle["warnings"]:
        document.add_paragraph(warning, style="List Bullet")
    document.add_paragraph(
        "Feature leakage requires domain review. Results describe these data partitions and do not guarantee future performance. "
        "Download this report and other artifacts before ending your session. Server-side session data is temporary."
    )
    document.add_heading("Execution evidence", level=1)
    for item in evidence:
        receipt = item["receipt"]
        document.add_paragraph(
            f"{item['task']}: {receipt['elapsed_seconds']} seconds; network disabled; "
            f"memory cap {receipt['memory_mb']} MiB; sandbox destroyed. "
            f"Code SHA-256: {receipt['code_sha256']}"
        )
    document.save(OUT / "report.docx")
    return {
        "report": "report.docx",
        "test_metrics": observed,
        "note": "Narrative generated by the agent; tables and plots computed from the frozen model.",
    }
