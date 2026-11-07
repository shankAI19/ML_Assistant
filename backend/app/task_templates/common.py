import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .paths import OUT


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(filename, value):
    (OUT / filename).write_text(json.dumps(clean(value), allow_nan=False), encoding="utf-8")


def load_csv(path):
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) > 100000 or len(frame.columns) > 200:
        raise ValueError("Demo limit: 100,000 rows and 200 columns. Please upload a smaller CSV.")
    if len(frame) < 1:
        raise ValueError("At least one row is needed")
    if any(len(str(c)) > 200 for c in frame.columns):
        raise ValueError("Column names must be at most 200 characters")
    return frame.replace([np.inf, -np.inf], np.nan)


def prepare(frame, plan):
    target = plan["target"]
    if target not in frame:
        raise ValueError("The target column does not exist")
    dropped = int(frame[target].isna().sum())
    frame = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    labels = frame[target].astype(str)
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2 or plan["positive_label"] not in classes:
        raise ValueError(f"Choose one of exactly two target values: {classes[:10]}")
    y = (labels == plan["positive_label"]).astype(int)
    if y.value_counts().min() < 10:
        raise ValueError("At least 10 examples of each class are needed for reliable splitting")
    unknown = set(plan["excluded_columns"]) - set(frame.columns)
    if unknown:
        raise ValueError(f"Excluded columns do not exist: {sorted(unknown)}")
    excluded = {target, *plan["excluded_columns"]}
    if plan["split_column"]:
        excluded.add(plan["split_column"])
    features = [c for c in frame.columns if c not in excluded]
    if not features:
        raise ValueError("No feature columns remain")
    return frame, y, features, classes, dropped


def split_indices(frame, y, plan):
    indices = np.arange(len(frame))
    test_fraction = plan.get("test_fraction", 0.2)
    val_fraction = plan.get("validation_fraction", 0.2)
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid train/validation/test fractions")
    test = np.array([], dtype=int)
    if plan["split"] == "stratified":
        dev = indices
        if test_fraction:
            dev, test = train_test_split(
                indices, test_size=test_fraction, stratify=y, random_state=plan["seed"]
            )
        train, val = train_test_split(
            dev,
            test_size=val_fraction / (1 - test_fraction),
            stratify=y.iloc[dev],
            random_state=plan["seed"],
        )
    elif plan["split"] == "chronological":
        times = pd.to_datetime(frame[plan["split_column"]], errors="coerce", utc=True)
        if times.isna().any():
            raise ValueError("The time column contains missing or invalid timestamps")
        order = np.argsort(times.to_numpy(), kind="stable")
        train, val, test = np.split(
            order,
            [
                int(len(order) * (1 - val_fraction - test_fraction)),
                int(len(order) * (1 - test_fraction)),
            ],
        )
        if not len(train) or not len(val) or (test_fraction and not len(test)):
            raise ValueError("Split fractions leave an empty partition")
        # Identical timestamps must not cross a partition boundary.
        if times.iloc[train].max() >= times.iloc[val].min() or (
            len(test) and times.iloc[val].max() >= times.iloc[test].min()
        ):
            raise ValueError(
                "Tied timestamps cross split boundaries; choose a dataset with a clear time boundary"
            )
    else:
        groups = frame[plan["split_column"]]
        if groups.isna().any():
            raise ValueError("The grouping column contains missing values")
        dev_local = indices
        if test_fraction:
            dev_local, test = next(
                GroupShuffleSplit(
                    n_splits=1, test_size=test_fraction, random_state=plan["seed"]
                ).split(indices, y, groups)
            )
        train_local, val_local = next(
            GroupShuffleSplit(
                n_splits=1, test_size=val_fraction / (1 - test_fraction), random_state=plan["seed"]
            ).split(dev_local, y.iloc[dev_local], groups.iloc[dev_local])
        )
        train, val = dev_local[train_local], dev_local[val_local]
    for name, subset in [("train", train), ("validation", val), ("test", test)]:
        if name == "test" and not test_fraction:
            continue
        if (
            len(subset) == 0
            or y.iloc[subset].nunique() != 2
            or y.iloc[subset].value_counts().min() < 2
        ):
            raise ValueError(
                f"The {name} split needs at least two examples of each class. Revisit the split or provide more data."
            )
    return train, val, test


def normalize(frame, features, numeric):
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
    result = frame[features].copy()
    for col in features:
        if col in numeric:
            converted = pd.to_numeric(result[col], errors="coerce")
            if (result[col].notna() & converted.isna()).any():
                raise ValueError(f"Column {col!r} contains nonnumeric values")
            result[col] = converted.replace([np.inf, -np.inf], np.nan)
        else:
            result[col] = result[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
    return result


def metrics(y, scores, threshold):
    predictions = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    result = {
        "rows": len(y),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "threshold": threshold,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0) if tp + fp else None,
        "recall": recall_score(y, predictions, zero_division=0) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": f1_score(y, predictions, zero_division=0) if 2 * tp + fp + fn else None,
        "roc_auc": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else None,
        "average_precision": average_precision_score(y, scores) if np.sum(y) else None,
        "alert_rate": float(predictions.mean()),
        "false_positives_per_1000": fp / len(y) * 1000,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return clean(result)


def charts(y, scores, threshold, include_thresholds=True):
    m = metrics(y, scores, threshold)
    definitions = {}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        definitions["roc"] = {
            "data": [
                {
                    "x": fpr.tolist(),
                    "y": tpr.tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Model",
                },
                {
                    "x": [0, 1],
                    "y": [0, 1],
                    "mode": "lines",
                    "name": "Chance",
                    "line": {"dash": "dot"},
                },
            ],
            "layout": {
                "title": {"text": "ROC curve"},
                "xaxis": {"title": {"text": "False positive rate"}},
                "yaxis": {"title": {"text": "Recall"}},
            },
        }
        definitions["precision_recall"] = {
            "data": [
                {"x": recall.tolist(), "y": precision.tolist(), "mode": "lines", "name": "Model"},
                {
                    "x": [0, 1],
                    "y": [m["prevalence"]] * 2,
                    "mode": "lines",
                    "name": "Prevalence",
                    "line": {"dash": "dot"},
                },
            ],
            "layout": {
                "title": {"text": "Precision–recall curve"},
                "xaxis": {"title": {"text": "Recall"}},
                "yaxis": {"title": {"text": "Precision"}},
            },
        }
    definitions["confusion"] = {
        "data": [
            {
                "type": "heatmap",
                "z": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                "x": ["Predicted negative", "Predicted positive"],
                "y": ["Actual negative", "Actual positive"],
                "colorscale": "Teal",
                "texttemplate": "%{z}",
            }
        ],
        "layout": {"title": {"text": "Confusion matrix"}, "yaxis": {"autorange": "reversed"}},
    }
    thresholds = np.linspace(0.01, 0.99, 51)
    summary = []
    for t in thresholds:
        selected = np.asarray(scores) >= t
        tp = int(np.sum(selected & (np.asarray(y) == 1)))
        fp = int(np.sum(selected & (np.asarray(y) == 0)))
        fn = int(np.sum(~selected & (np.asarray(y) == 1)))
        summary.append(
            {
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                "alert_rate": float(selected.mean()),
            }
        )
    if include_thresholds:
        definitions["thresholds"] = {
            "data": [
                {
                    "x": thresholds.tolist(),
                    "y": [v[k] for v in summary],
                    "mode": "lines",
                    "name": k.replace("_", " "),
                }
                for k in ["precision", "recall", "f1", "alert_rate"]
            ],
            "layout": {
                "title": {"text": "Threshold tradeoffs (read-only)"},
                "xaxis": {"title": {"text": "Decision threshold"}},
            },
        }
    for name, figure in definitions.items():
        save_json(f"{name}.json", figure)
    return [f"{name}.json" for name in definitions]


def safe_csv_label(value):
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text
