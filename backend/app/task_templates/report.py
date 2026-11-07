"""Produce scientific reports from saved results without rerunning ML."""

import json
import re
from zipfile import ZIP_DEFLATED, ZipFile

from .paths import OUT, WORK
from .report_content import MODELS, evidence_markdown
from .report_document import to_docx


def saved_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence = json.loads((WORK / "evidence.json").read_text(encoding="utf-8"))
    contexts = {}
    training_round = uploaded = 0
    for index, item in enumerate(evidence, 1):
        task, result = item["task"], item["result"]
        if task == "train":
            training_round += 1
            model = MODELS.get(result.get("model"), "classifier")
            context = f"training round {training_round} validation data ({model})"
        elif task == "evaluate":
            context = f"the held-out test data for training round {training_round}"
        elif task == "predict":
            uploaded += 1
            context = f"uploaded labeled dataset {uploaded}"
        else:
            context = "the recorded assessment"
        contexts[item.get("sequence", index)] = (context, result)
    plots = []
    for source in sorted(WORK.glob("chart_*.json"))[-6:]:
        figure = None
        try:
            match = re.fullmatch(r"chart_(\d+)_(train|evaluate|predict)_(.+)\.json", source.name)
            if not match:
                continue
            context, result = contexts.get(int(match[1]) + 1, ("the recorded assessment", {}))
            kind = match[3]
            titles = {
                "confusion": "Confusion matrix",
                "roc": "Receiver operating characteristic",
                "precision_recall": "Precision and recall",
                "thresholds": "Decision threshold tradeoffs",
            }
            title = titles.get(kind, "Recorded model assessment")
            definition = json.loads(source.read_text(encoding="utf-8"))
            figure, axis = plt.subplots(figsize=(7.2, 4.3))
            for trace in definition["data"]:
                if trace.get("type") == "heatmap":
                    values = trace["z"]
                    image = axis.imshow(values, cmap="Blues", aspect="auto")
                    maximum = max(max(row) for row in values)
                    for r, row in enumerate(values):
                        for c, value in enumerate(row):
                            axis.text(
                                c,
                                r,
                                str(value),
                                ha="center",
                                va="center",
                                color="white" if value > maximum / 2 else "black",
                            )
                    axis.set_xticks(range(len(trace["x"])), ["Negative", "Positive"])
                    axis.set_yticks(range(len(trace["y"])), ["Negative", "Positive"])
                    axis.set_xlabel("Predicted class")
                    axis.set_ylabel("Observed class")
                    figure.colorbar(image, ax=axis, label="Number of cases")
                else:
                    draw = axis.step if kind == "precision_recall" else axis.plot
                    options = {"where": "post"} if kind == "precision_recall" else {}
                    axis_options = (
                        {"linestyle": "--"} if trace.get("line", {}).get("dash") == "dot" else {}
                    )
                    draw(
                        trace["x"],
                        trace["y"],
                        label=trace.get("name", "").replace("_", " ").capitalize(),
                        linewidth=1.8,
                        **options,
                        **axis_options,
                    )
            layout = definition.get("layout", {})
            if kind != "confusion":
                axis.set_xlabel(layout.get("xaxis", {}).get("title", {}).get("text", "Score"))
                axis.set_ylabel(
                    layout.get("yaxis", {})
                    .get("title", {})
                    .get("text", "Metric value or fraction flagged")
                )
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1.03)
                axis.grid(alpha=0.2)
            if kind == "thresholds":
                threshold = result.get("plan", {}).get("threshold")
                if threshold is not None:
                    axis.axvline(
                        threshold, color="#555555", linestyle=":", label="Selected threshold"
                    )
            if axis.lines:
                axis.legend(frameon=False, fontsize=9)
            axis.set_title(title, fontsize=12, pad=12)
            figure.tight_layout()
            path = OUT / f"figure_{len(plots) + 1:02}.png"
            figure.savefig(path, dpi=170, facecolor="white")
            description = {
                "confusion": "Rows show observed classes and columns show predicted classes at the recorded decision threshold. Cell counts show correct classifications, false alarms, and missed positive cases.",
                "roc": "Recall is plotted against the false-positive rate across score thresholds. The diagonal represents random ranking.",
                "precision_recall": "Precision is plotted against recall across score thresholds. The prevalence reference shows the positive-class frequency in the assessed data.",
                "thresholds": "Precision, recall, F1, and the fraction flagged are shown across decision thresholds. Changing a threshold trades missed positives against false alarms and review volume; this plot does not constitute a new training round.",
            }.get(kind, "This illustration was rendered from the saved assessment.")
            plots.append(
                {"path": path, "title": title, "caption": f"{title} for {context}. {description}"}
            )
        except Exception:  # noqa: BLE001, S112 - caller reports any omitted saved figures
            continue
        finally:
            if figure is not None:
                plt.close(figure)
    return plots


def run(payload):
    evidence = json.loads((WORK / "evidence.json").read_text(encoding="utf-8"))
    result = {
        "markdown": "report.md",
        "report": None,
        "bundle": "report.zip",
        "figures": [],
        "note": "Scientific report of completed work, with methods, results tables, interpretation, and saved figures. No training or new evaluation was performed.",
        "warnings": [],
    }
    try:
        plots = saved_plots()
        if len(plots) < len(list(WORK.glob("chart_*.json"))[-6:]):
            result["warnings"].append(
                "Some saved figures could not be rendered. The recorded numerical results remain available in the tables."
            )
    except Exception:  # noqa: BLE001 - tables and DOCX remain useful if plotting is unavailable
        plots = []
        result["warnings"].append(
            "Saved figures could not be rendered. The recorded numerical results remain available in the tables."
        )
    markdown = evidence_markdown(evidence, payload.get("narrative", ""), plots)
    if result["warnings"]:
        markdown += "\n## Figure availability\n\n" + "\n\n".join(result["warnings"]) + "\n"
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    result["figures"] = [plot["path"].name for plot in plots]
    try:
        to_docx(markdown, plots)
        result["report"] = "report.docx"
    except Exception:  # noqa: BLE001 - preserve the illustrated Markdown bundle on conversion failure
        (OUT / "report.docx").unlink(missing_ok=True)
        result["warnings"].append(
            "DOCX conversion could not complete. Download the illustrated report ZIP or Markdown report instead."
        )
    with ZipFile(OUT / "report.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(OUT / "report.md", "report.md")
        if result["report"]:
            archive.write(OUT / "report.docx", "report.docx")
        for plot in plots:
            archive.write(plot["path"], plot["path"].name)
    return result
