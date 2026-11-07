"""Build reports from saved evidence only, without loading or evaluating a model."""

import json
from pathlib import Path

OUT = Path("/work/out")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def evidence_markdown(evidence, narrative):
    lines = ["# Your classification report", "", "Work completed in this temporary session.", ""]
    if narrative.strip():
        lines += ["## Interpretation", "", narrative.strip(), ""]
    if not evidence:
        lines += ["No analysis or training has been completed yet.", ""]
    for index, item in enumerate(evidence, 1):
        result = item["result"]
        lines += [f"## {index}. {item['task'].title()}", ""]
        if item["task"] == "profile":
            lines += [
                f"Rows: {result.get('rows', 'unknown')}. Duplicates: {result.get('duplicates', 'unknown')}.",
                "",
                "| Column | Type | Missing | Unique |",
                "| --- | --- | --- | --- |",
            ]
            for column in result.get("columns", []):
                lines.append(
                    "| "
                    + " | ".join(
                        cell(column.get(k, "")) for k in ("name", "dtype", "missing", "unique")
                    )
                    + " |"
                )
            lines.append("")
        if result.get("plan"):
            lines += [
                "### Approved training recipe",
                "",
                "```json",
                json.dumps(result["plan"], indent=2),
                "```",
                "",
                "Learned preprocessing was fitted on training data only.",
                "",
            ]
        if result.get("features"):
            lines += ["Features: " + ", ".join(cell(v) for v in result["features"]), ""]
        if result.get("split_counts"):
            lines += ["Actual partition sizes: " + json.dumps(result["split_counts"]), ""]
        for label, key in [
            ("Training", "training_metrics"),
            ("Validation", "validation_metrics"),
            ("Test / uploaded labeled data", "metrics"),
        ]:
            values = result.get(key)
            if isinstance(values, dict):
                lines += [f"### {label} metrics", "", "| Metric | Value |", "| --- | --- |"]
                for name, value in values.items():
                    display = (
                        "Unavailable"
                        if value is None
                        else f"{value:.4f}"
                        if isinstance(value, float)
                        else str(value)
                    )
                    lines.append(f"| {cell(name.replace('_', ' '))} | {display} |")
                lines.append("")
        if item["task"] == "predict":
            lines += [
                f"Prediction rows: {result.get('rows')}. Alert rate: {result.get('alert_rate')}.",
                "",
            ]
        if result.get("note"):
            lines += [result["note"], ""]
        for warning in result.get("warnings", []):
            lines.append(f"- {warning}")
        lines += [
            "",
            "Execution evidence: " + json.dumps(item.get("receipt", {}), sort_keys=True),
            "",
        ]
    tasks = {item["task"] for item in evidence}
    lines += ["## Work remaining and limitations", ""]
    if "train" not in tasks:
        lines.append("- No model has been trained.")
    if "evaluate" not in tasks:
        lines.append("- The model has not been finalized.")
    if not any(isinstance(item["result"].get("metrics"), dict) for item in evidence):
        lines.append(
            "- No test performance has been measured. Validation describes development performance only."
        )
    lines += [
        "- Unlabeled data supports predictions, not label-dependent performance metrics.",
        "- Scores are uncalibrated unless calibration was explicitly assessed. Average precision is not trapezoidal PR area.",
        "- Leakage requires domain review; observed results do not guarantee future performance.",
        "- Download outputs before ending the session. All session data is temporary.",
        "",
    ]
    return "\n".join(lines)


def add_inline(paragraph, token):
    bold = italic = False
    for child in token.children or []:
        if child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type in {"text", "code_inline"}:
            run = paragraph.add_run(child.content)
            run.bold, run.italic = bold, italic
            if child.type == "code_inline":
                run.font.name = "Consolas"
        elif child.type in {"softbreak", "hardbreak"}:
            paragraph.add_run("\n")


def to_docx(markdown, plots):
    from docx import Document
    from docx.shared import Inches, Pt
    from markdown_it import MarkdownIt

    document = Document()
    document.styles["Normal"].font.size = Pt(11)
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    paragraph = table = row = None
    list_style = None
    column = 0
    for token in tokens:
        kind = token.type
        if kind == "heading_open":
            paragraph = document.add_heading(level=min(int(token.tag[1:]), 4))
        elif kind == "bullet_list_open":
            list_style = "List Bullet"
        elif kind == "ordered_list_open":
            list_style = "List Number"
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            list_style = None
        elif kind == "paragraph_open":
            paragraph = document.add_paragraph(style=list_style)
        elif kind == "table_open":
            table = document.add_table(rows=0, cols=0)
            table.style = "Light Shading Accent 1"
        elif kind == "tr_open":
            row, column = table.add_row(), 0
        elif kind in {"th_open", "td_open"}:
            if column >= len(table.columns):
                table.add_column(Inches(2.5))
            paragraph = row.cells[column].paragraphs[0]
            column += 1
        elif kind == "table_close":
            width = Inches(6 / max(1, len(table.columns)))
            for grid_column in table.columns:
                grid_column.width = width
            for table_row in table.rows:
                for table_cell in table_row.cells:
                    table_cell.width = width
        elif kind == "inline":
            if paragraph is None:
                paragraph = document.add_paragraph()
            add_inline(paragraph, token)
        elif kind in {"fence", "code_block"}:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(token.content)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
    if plots:
        document.add_heading("Saved evaluation plots", level=1)
        for path in plots:
            document.add_picture(str(path), width=Inches(5.8))
    document.save(OUT / "report.docx")


def saved_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for source in sorted(Path("/work").glob("chart_*.json"))[-6:]:
        figure = None
        try:
            definition = json.loads(source.read_text(encoding="utf-8"))
            figure, axis = plt.subplots(figsize=(7, 4))
            for trace in definition["data"]:
                if trace.get("type") == "heatmap":
                    axis.imshow(trace["z"], cmap="Blues")
                    for r, values in enumerate(trace["z"]):
                        for c, value in enumerate(values):
                            axis.text(c, r, str(value), ha="center", va="center")
                    axis.set_xticks(range(len(trace["x"])), trace["x"])
                    axis.set_yticks(range(len(trace["y"])), trace["y"])
                else:
                    axis.plot(trace["x"], trace["y"], label=trace.get("name", ""))
            layout = definition.get("layout", {})
            stage = {
                "train": "Validation",
                "evaluate": "Held-out test",
                "predict": "Uploaded labeled data",
            }.get(source.name.split("_")[2], "Saved results")
            axis.set_title(f"{stage}: {layout.get('title', {}).get('text', source.stem)}")
            axis.set_xlabel(layout.get("xaxis", {}).get("title", {}).get("text", ""))
            axis.set_ylabel(layout.get("yaxis", {}).get("title", {}).get("text", ""))
            if axis.lines:
                axis.legend()
            figure.tight_layout()
            path = OUT / f"saved_{len(paths)}.png"
            figure.savefig(path, dpi=120)
            paths.append(path)
        except Exception:  # noqa: BLE001, S110 - a malformed saved chart must not prevent a report
            pass
        finally:
            if figure is not None:
                plt.close(figure)
    return paths


def run(payload):
    evidence = json.loads(Path("/work/evidence.json").read_text(encoding="utf-8"))
    markdown = evidence_markdown(evidence, payload.get("narrative", ""))
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    result = {
        "markdown": "report.md",
        "report": None,
        "note": "Report summarizes saved evidence only; no training or new evaluation was performed.",
        "warnings": [],
    }
    try:
        to_docx(markdown, saved_plots())
        result["report"] = "report.docx"
    except Exception:  # noqa: BLE001 - preserve the Markdown download on conversion failure
        (OUT / "report.docx").unlink(missing_ok=True)
        result["warnings"].append(
            "DOCX conversion could not complete. Download the complete Markdown report instead."
        )
    return result
