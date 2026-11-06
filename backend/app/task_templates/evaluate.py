import joblib

from .common import charts, load_csv, metrics, normalize, prepare, split_indices


def evaluate_bundle():
    bundle = joblib.load("/work/model.joblib")
    frame, y, _, _, _ = prepare(load_csv("/work/input.csv"), bundle["plan"])
    _, _, test = split_indices(frame, y, bundle["plan"])
    x = normalize(frame, bundle["features"], bundle["numeric"])
    scores = bundle["pipeline"].predict_proba(x.iloc[test])[:, 1]
    return bundle, y.iloc[test], scores


def run(payload):
    bundle, y, scores = evaluate_bundle()
    return {
        "evaluation_partition": "held-out test",
        "model": bundle["plan"]["model"],
        "metrics": metrics(y, scores, bundle["plan"]["threshold"]),
        "charts": charts(y, scores, bundle["plan"]["threshold"], include_thresholds=False),
        "warnings": bundle["warnings"],
        "note": "This exact fitted model and threshold are frozen. No full-data refit was performed.",
    }
