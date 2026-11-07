"""Human-readable scientific reporting of recorded experiment evidence."""

import ast
import json
import re

MODELS = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
METRICS = {
    "average_precision": "Average precision",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1 score",
    "roc_auc": "ROC AUC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "alert_rate": "Alert rate",
    "false_positives_per_1000": "False alarms per 1,000 rows",
}
RATES = {"precision", "recall", "specificity", "accuracy", "balanced_accuracy", "alert_rate"}


def cell(value):
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    )


def table(lines, headings, rows):
    lines += [
        "| " + " | ".join(map(cell, headings)) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def number(value):
    return (
        f"{value:,}" if isinstance(value, int) else "Not recorded" if value is None else str(value)
    )


def metric(key, values):
    if key not in values:
        return "Not recorded"
    value = values[key]
    if value is None:
        return "Not estimable"
    if key in RATES or key == "prevalence":
        return f"{value:.1%}"
    return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)


def proportion(value):
    return f"{value * 100:.4f}".rstrip("0").rstrip(".") + "%"


def readable_narrative(narrative):
    """Exclude configuration dumps; preserve short explanatory Python examples."""
    text = narrative.strip()
    try:
        if isinstance(json.loads(text), (dict, list)):
            return ""
    except (ValueError, TypeError):
        pass

    def code_block(match):
        language, body = match.group(1).strip().lower(), match.group(2)
        if language in {"python", "py"} and len(body.splitlines()) <= 20:
            return f"**Illustrative Python example**\n\n```python\n{body.rstrip()}\n```"
        return ""

    text = re.sub(r"```([^\n]*)\n(.*?)```", code_block, text, flags=re.DOTALL).strip()
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = paragraph.strip()
        if candidate.startswith(("{", "[")):
            parsed = None
            for parse in (json.loads, ast.literal_eval):
                try:
                    parsed = parse(candidate)
                    break
                except (ValueError, SyntaxError):
                    pass
            if isinstance(parsed, (dict, list)):
                continue
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def study_summary(evidence, plans):
    if not evidence:
        return "No analysis or training has been completed yet. Upload and inspect a labeled dataset to begin the study."
    if not plans:
        return "The dataset has been inspected. No model has been trained, so predictive performance has not been established."
    for item in reversed(evidence):
        result = item["result"]
        values = result.get("metrics") or result.get("validation_metrics")
        if isinstance(values, dict):
            stage = {
                "train": "validation",
                "evaluate": "held-out test",
                "predict": "uploaded labeled data",
            }.get(item["task"], "recorded")
            sentence = f"The latest {stage} assessment is reported below"
            if values.get("rows") is not None:
                sentence += f" for {number(values['rows'])} rows"
            sentence += "."
            if values.get("precision") is not None and values.get("recall") is not None:
                sentence += f" Precision was {metric('precision', values)} and recall was {metric('recall', values)}."
            return (
                sentence
                + " These estimates describe the assessed data; they do not establish future operational performance."
            )
    return "A training round has been recorded, but usable performance estimates are not available in this report."


def data_section(lines, profiles, plan):
    lines += ["## Data and quality assessment", ""]
    if not profiles:
        lines += ["A dataset quality assessment has not been recorded.", ""]
        return
    profile = profiles[-1]["result"]
    columns, rows = profile.get("columns", []), profile.get("rows")
    lines += [
        f"The uploaded dataset contains {number(rows)} rows and {len(columns)} columns. The inspection identified {number(profile.get('duplicates'))} exact duplicate rows.",
        "",
    ]
    quality = []
    for column in columns:
        missing = column.get("missing")
        missing_text = number(missing)
        if rows and isinstance(missing, int):
            missing_text += f" ({missing / rows:.1%})"
        dtype = str(column.get("dtype", "Not recorded"))
        kind = (
            "Numeric"
            if dtype.startswith(("int", "float", "uint"))
            else "Boolean"
            if dtype == "bool"
            else "Date and time"
            if dtype.startswith("datetime")
            else "Text or categorical"
            if dtype in {"object", "string", "category"}
            else dtype
        )
        notes = []
        if column.get("name") == plan.get("target"):
            notes.append("Prediction target")
        if column.get("possible_id"):
            notes.append("Possible identifier")
        quality.append(
            (
                column.get("name", "Unnamed"),
                kind,
                missing_text,
                number(column.get("unique")),
                "; ".join(notes) or "None",
            )
        )
    table(
        lines,
        ["Column", "Data type", "Missing values", "Distinct values", "Role or observation"],
        quality,
    )
    target = next((column for column in columns if column.get("name") == plan.get("target")), None)
    if target and target.get("value_counts"):
        lines += [
            "### Target class distribution",
            "",
            "Counts describe the uploaded data before training exclusions and duplicate removal.",
            "",
        ]
        table(
            lines,
            ["Target label", "Count", "Role"],
            [
                (
                    label,
                    number(count),
                    "Positive class"
                    if str(label) == str(plan.get("positive_label"))
                    else "Negative class",
                )
                for label, count in target["value_counts"].items()
            ],
        )
    lines += [
        "Missing values, identifiers, and the timing of predictors require review before modeling. A data profile alone cannot establish that predictors are free of information leakage.",
        "",
    ]


def methods_section(lines, training):
    lines += ["## Methods", ""]
    if not training:
        lines += [
            "No training method has been executed. Model selection, predictors, and evaluation design remain to be agreed.",
            "",
        ]
        return
    for index, item in enumerate(training, 1):
        result, plan = item["result"], item["result"].get("plan", {})
        lines += [f"### Training round {index}", ""]
        if not plan:
            lines += ["The complete training configuration was not recorded.", ""]
            continue
        model = MODELS.get(plan.get("model"), "Not recorded")
        split = {
            "stratified": "Stratified random split",
            "group": "Split by groups",
            "chronological": "Chronological split",
        }.get(plan.get("split"), "Not recorded")
        settings = [
            ("Classifier", model),
            ("Target column", plan.get("target", "Not recorded")),
            ("Positive class label", plan.get("positive_label", "Not recorded")),
            ("Decision threshold", number(plan.get("threshold"))),
            ("Partitioning strategy", split),
            ("Random seed", number(plan.get("seed"))),
            ("Class imbalance weighting", "Enabled" if plan.get("class_weighted") else "Disabled"),
        ]
        if plan.get("split_column"):
            settings.append(("Grouping or time column", plan["split_column"]))
        if plan.get("model") == "logistic":
            settings.append(
                ("Inverse regularization strength C", number(plan.get("regularization")))
            )
        else:
            settings += [
                ("Number of trees", number(plan.get("trees"))),
                ("Maximum tree depth", number(plan.get("max_depth"))),
            ]
            if plan.get("model") in {"xgboost", "lightgbm"}:
                settings.append(("Learning rate", number(plan.get("learning_rate"))))
        table(lines, ["Method setting", "Recorded value"], settings)
        features = result.get("features")
        if features:
            lines += [
                "Included predictors: " + ", ".join(cell(value) for value in features) + ".",
                "",
            ]
        excluded = plan.get("excluded_columns", [])
        lines += [
            "Explicitly excluded columns: "
            + (", ".join(cell(value) for value in excluded) if excluded else "none")
            + ".",
            "",
        ]
        lines += [
            "The preprocessing pipeline uses training medians for numeric missing values and a missing category followed by one-hot encoding for categorical predictors. Learned preprocessing was fitted on training data only.",
            "",
        ]
        if plan.get("model") == "logistic":
            lines += ["Numeric predictors were standardized for logistic regression.", ""]
            expansion = plan.get("feature_expansion", "none")
            if expansion != "none":
                description = (
                    "interaction terms only"
                    if expansion == "interaction_only"
                    else "polynomial and interaction terms"
                )
                lines += [
                    f"Numeric feature expansion used {description} up to degree {plan.get('polynomial_degree', 2)}. The recorded expanded numeric feature count was {number(result.get('numeric_features_after_expansion'))}.",
                    "",
                ]
        counts = result.get("split_counts", {})
        val, test = plan.get("validation_fraction", 0.2), plan.get("test_fraction", 0.2)
        table(
            lines,
            ["Partition", "Requested proportion", "Recorded rows", "Purpose"],
            [
                (
                    "Training",
                    proportion(1 - val - test),
                    number(counts.get("train")),
                    "Fit preprocessing and classifier",
                ),
                (
                    "Validation",
                    proportion(val),
                    number(counts.get("validation")),
                    "Guide development decisions",
                ),
                (
                    "Held-out test",
                    proportion(test),
                    number(counts.get("test")),
                    "Final assessment after finalization"
                    if test
                    else "No internal test set reserved",
                ),
            ],
        )
        if plan.get("split") == "group":
            lines += [
                "Requested proportions apply to groups; actual row proportions can differ.",
                "",
            ]


def assessment(lines, result, partitions):
    available = [
        (label, result[key]) for label, key in partitions if isinstance(result.get(key), dict)
    ]
    if not available:
        return
    table(
        lines,
        ["Metric"] + [label for label, _ in available],
        [
            (label, *(metric(key, values) for _, values in available))
            for key, label in METRICS.items()
            if any(key in values for _, values in available)
        ],
    )
    table(
        lines,
        ["Assessment data", "Rows", "Positive cases", "Prevalence", "Threshold"],
        [
            (
                label,
                number(values.get("rows")),
                number(values.get("positives")),
                metric("prevalence", values),
                number(values.get("threshold")),
            )
            for label, values in available
        ],
    )
    label, values = available[-1]
    if all(values.get(key) is not None for key in ("tp", "fn", "fp", "tn")):
        lines += [
            f"On the {label.lower()} data, {number(values['tp'])} positive cases were correctly flagged and {number(values['fn'])} were missed. There were {number(values['fp'])} false alarms among negative cases.",
            "",
        ]
        table(
            lines,
            ["Observed class", "Predicted negative", "Predicted positive"],
            [
                ("Negative", number(values["tn"]), number(values["fp"])),
                ("Positive", number(values["fn"]), number(values["tp"])),
            ],
        )


def results_section(lines, evidence):
    lines += ["## Results", ""]
    round_number = uploaded_number = 0
    any_assessment = False
    for item in evidence:
        result, task = item["result"], item["task"]
        if task == "train":
            round_number += 1
            lines += [
                f"### Training round {round_number}",
                "",
                "Training estimates describe the fitted data. Validation estimates describe development performance on a separate partition.",
                "",
            ]
            assessment(
                lines,
                result,
                [("Training", "training_metrics"), ("Validation", "validation_metrics")],
            )
            baseline = result.get("baseline", {})
            if baseline:
                lines += [
                    "The following simple reference predictions were assessed on the validation partition.",
                    "",
                ]
                table(
                    lines,
                    ["Reference prediction", "Metric", "Value"],
                    [
                        (
                            "Always predict the negative class",
                            "Accuracy",
                            metric(
                                "accuracy", {"accuracy": baseline.get("accuracy_always_negative")}
                            ),
                        ),
                        (
                            "Always predict the negative class",
                            "Recall",
                            metric("recall", {"recall": baseline.get("recall_always_negative")}),
                        ),
                        (
                            "Assign every row the same score",
                            "Average precision",
                            metric(
                                "average_precision",
                                {
                                    "average_precision": baseline.get(
                                        "constant_score_average_precision"
                                    )
                                },
                            ),
                        ),
                    ],
                )
                lines += [
                    "The constant-score average precision equals positive-class prevalence. Accuracy alone may conceal failure to detect positive cases.",
                    "",
                ]
            any_assessment = True
        elif task == "evaluate":
            lines += ["### Finalization and held out test assessment", ""]
            if isinstance(result.get("metrics"), dict):
                assessment(lines, result, [("Held-out test", "metrics")])
            else:
                lines += [
                    "The fitted model was finalized without an internal test assessment. No test-performance estimate is available from this step.",
                    "",
                ]
            lines += [
                "Finalization retains the assessed fitted pipeline and threshold; it does not refit the model on all rows.",
                "",
            ]
            any_assessment = True
        elif task == "predict":
            uploaded_number += 1
            lines += [
                f"### Uploaded data assessment {uploaded_number}",
                "",
                f"Predictions were generated for {number(result.get('rows'))} uploaded rows. The alert rate was {metric('alert_rate', result)}; this is the proportion flagged positive, not a measure of correctness.",
                "",
            ]
            if isinstance(result.get("metrics"), dict):
                assessment(lines, result, [("Uploaded labeled data", "metrics")])
            else:
                lines += [
                    "The uploaded data had no outcome labels. Accuracy, precision, recall, and other label-dependent performance measures cannot be established for these predictions.",
                    "",
                ]
            any_assessment = True
    if not any_assessment:
        lines += ["No predictive performance has been measured.", ""]
    else:
        lines += [
            "Not estimable indicates an undefined metric, such as precision when no cases are flagged. Not recorded indicates a value absent from the saved assessment.",
            "",
        ]


def evidence_markdown(evidence, narrative, plots=()):
    training = [item for item in evidence if item["task"] == "train"]
    profiles = [item for item in evidence if item["task"] == "profile"]
    plan = training[-1]["result"].get("plan", {}) if training else {}
    lines = [
        "# Binary classification study report",
        "",
        "## Summary",
        "",
        study_summary(evidence, training),
        "",
    ]
    if plan.get("target"):
        lines += [
            f"The prediction target was {cell(plan['target'])}, with {cell(plan.get('positive_label', 'not recorded'))} designated as the positive class.",
            "",
        ]
    interpretation = readable_narrative(narrative)
    if interpretation:
        lines += ["## Study objective and interpretation", "", interpretation, ""]
    data_section(lines, profiles, plan)
    methods_section(lines, training)
    results_section(lines, evidence)
    if plots:
        lines += [
            "## Figures",
            "",
            "Figures summarize saved assessments from the most recent recorded tasks. No model was retrained to produce these illustrations.",
            "",
        ]
        for index, plot in enumerate(plots, 1):
            lines += [
                f"![{plot['title']}]({plot['path'].name})",
                "",
                f"*Figure {index}. {plot['caption']}*",
                "",
            ]
    lines += [
        "## Interpretation of the metrics",
        "",
        "Precision is the fraction of flagged cases that are positive. Recall is the fraction of positive cases detected; specificity is the fraction of negative cases correctly rejected. F1 summarizes precision and recall. The alert rate estimates review volume rather than correctness.",
        "",
        "The area under the receiver operating characteristic curve (ROC AUC) summarizes ranking across false-positive rates. Average precision summarizes the precision-recall curve and should be interpreted relative to positive-class prevalence; it is not the trapezoidal area under that curve. Neither measure selects an operational decision threshold.",
        "",
        "## Limitations and next steps",
        "",
    ]
    tasks = {item["task"] for item in evidence}
    if not training:
        lines += [
            "No model has been trained. Agree on the target, predictors, and partitioning design before training.",
            "",
        ]
    if "evaluate" not in tasks:
        lines += [
            "The model has not been finalized. Complete development decisions before finalization.",
            "",
        ]
    if not any(isinstance(item["result"].get("metrics"), dict) for item in evidence):
        lines += [
            "No test performance has been measured. Validation describes development performance only.",
            "",
        ]
    warnings = list(
        dict.fromkeys(
            warning for item in evidence for warning in item["result"].get("warnings", [])
        )
    )
    lines += [warning + "\n" for warning in warnings]
    lines += [
        "Scores should not be treated as calibrated probabilities. Repeated validation-based refinement can overfit development data. Uncertainty intervals and calibration analyses are not provided by this workflow. Small positive-class counts may make estimates unstable.",
        "",
        "Review the timing and availability of predictors, repeated entities, and possible information leakage. Performance on uploaded labeled data depends on whether those rows represent future use and were independent of model development. Business costs and review capacity should guide the choice of threshold.",
        "",
        "## Reproducibility and execution",
        "",
        "Results were collected from isolated task executions. Each training round used an approved plan. Reporting summarized existing evidence without training a model or repeating an assessment.",
        "",
    ]
    records = []
    for index, item in enumerate(evidence, 1):
        receipt = item.get("receipt", {})
        duration = receipt.get("elapsed_seconds")
        records.append(
            (
                f"{index}. { {'profile': 'Data inspection', 'train': 'Model training', 'evaluate': 'Finalization', 'predict': 'Prediction'}.get(item['task'], item['task']) }",
                f"{duration:.2f}" if isinstance(duration, (int, float)) else "Not recorded",
                "Disabled" if receipt.get("network") == "none" else "Not recorded",
                "Yes" if receipt.get("sandbox_destroyed") is True else "Not recorded",
            )
        )
    if records:
        table(
            lines,
            ["Recorded task", "Elapsed seconds", "Network access", "Container removed"],
            records,
        )
    lines += [
        "Exact scripts, standalone experiment packages, and machine-readable execution receipts are available separately from the application. Download the outputs before ending the temporary session.",
        "",
    ]
    return "\n".join(lines)
