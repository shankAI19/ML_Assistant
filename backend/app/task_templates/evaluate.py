import joblib

from .common import charts, load_csv, metrics, normalize, prepare, split_indices
from .paths import WORK


def evaluate_bundle():
    bundle = joblib.load(WORK / "model.joblib")
    frame, y, _, _, _ = prepare(load_csv(WORK / "input.csv"), bundle["plan"])
    _, _, test = split_indices(frame, y, bundle["plan"])
    x = normalize(frame, bundle["features"], bundle["numeric"])
    scores = bundle["pipeline"].predict_proba(x.iloc[test])[:, 1]
    return bundle, y.iloc[test], scores


def run(payload):
    bundle = joblib.load(WORK / "model.joblib")
    if not bundle["plan"].get("test_fraction", 0.2):
        return {
            "model": bundle["plan"]["model"],
            "metrics": None,
            "charts": [],
            "warnings": bundle["warnings"],
            "note": "Model finalized without an internal test set. Upload independent labeled data for test performance; unlabeled data supports predictions only.",
        }
    bundle, y, scores = evaluate_bundle()
    return {
        "evaluation_partition": "held-out test",
        "model": bundle["plan"]["model"],
        "metrics": metrics(y, scores, bundle["plan"]["threshold"]),
        "charts": charts(y, scores, bundle["plan"]["threshold"], include_thresholds=False),
        "warnings": bundle["warnings"],
        "note": "This exact fitted model and threshold are frozen. No full-data refit was performed.",
    }
