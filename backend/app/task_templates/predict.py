import joblib
import pandas as pd

from .common import charts, load_csv, metrics, normalize, safe_csv_label
from .paths import OUT, WORK


def run(payload):
    bundle = joblib.load(WORK / "model.joblib")
    frame = load_csv(WORK / "prediction.csv")
    x = normalize(frame, bundle["features"], bundle["numeric"])
    scores = bundle["pipeline"].predict_proba(x)[:, 1]
    plan = bundle["plan"]
    predicted = scores >= plan["threshold"]
    result = {
        "rows": len(frame),
        "alert_rate": float(predicted.mean()),
        "metrics": None,
        "charts": [],
        "warnings": bundle["warnings"],
        "note": "No performance metrics without true labels. Scores are not calibrated.",
    }
    labels = [plan["positive_label"] if value else bundle["negative_label"] for value in predicted]
    output = pd.DataFrame(
        {
            "row_id": range(len(frame)),
            "positive_class_score": scores,
            "predicted_label": [safe_csv_label(v) for v in labels],
            "threshold": plan["threshold"],
        }
    )
    if plan["target"] in frame:
        actual = frame[plan["target"]]
        if actual.isna().any() or not set(actual.astype(str).unique()).issubset(bundle["classes"]):
            raise ValueError("Uploaded labels contain missing or unknown values")
        y = (actual.astype(str) == plan["positive_label"]).astype(int)
        result["metrics"] = metrics(y, scores, plan["threshold"])
        result["charts"] = charts(y, scores, plan["threshold"], include_thresholds=False)
        result["note"] = (
            "Performance measured using labels in the uploaded prediction CSV. row_id is the zero-based original CSV row order."
        )
        output["actual_label"] = actual.map(safe_csv_label)
    output.to_csv(OUT / "predictions.csv", index=False)
    return result
