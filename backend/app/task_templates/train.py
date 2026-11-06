import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import OUT, charts, load_csv, metrics, normalize, prepare, split_indices


def run(payload):
    plan = payload["plan"]
    source = load_csv("/work/input.csv")
    frame, y, features, classes, dropped = prepare(source, plan)
    removed_duplicates = len(source) - dropped - len(frame)
    train, val, test = split_indices(frame, y, plan)
    numeric = [c for c in features if frame[c].dtype.kind in "biufc"]
    categorical = [c for c in features if c not in numeric]
    if any(frame.iloc[train][c].nunique() > 100 for c in categorical):
        raise ValueError(
            "A categorical column has more than 100 values; exclude IDs/high-cardinality features for this demo"
        )
    if len(categorical) > 50:
        raise ValueError("Demo limit: 50 categorical features")
    x = normalize(frame, features, numeric)
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if plan["model"] == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="missing", keep_empty_features=True
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    weights = "balanced" if plan["class_weighted"] else None
    ratio = (
        float((y.iloc[train] == 0).sum() / (y.iloc[train] == 1).sum())
        if plan["class_weighted"]
        else 1.0
    )
    if plan["model"] == "logistic":
        estimator = LogisticRegression(
            C=plan["regularization"], class_weight=weights, max_iter=1000, random_state=plan["seed"]
        )
    elif plan["model"] == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=plan["trees"],
            max_depth=plan["max_depth"],
            class_weight=weights,
            n_jobs=1,
            random_state=plan["seed"],
        )
    elif plan["model"] == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=plan["trees"],
            max_depth=plan["max_depth"],
            learning_rate=plan["learning_rate"],
            scale_pos_weight=ratio,
            n_jobs=1,
            random_state=plan["seed"],
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=plan["trees"],
            max_depth=plan["max_depth"],
            num_leaves=min(31, 2 ** plan["max_depth"]),
            learning_rate=plan["learning_rate"],
            scale_pos_weight=ratio,
            n_jobs=1,
            random_state=plan["seed"],
            verbosity=-1,
        )
    pipeline = Pipeline([("preprocess", prep), ("classifier", estimator)])
    pipeline.fit(x.iloc[train], y.iloc[train])
    scores = pipeline.predict_proba(x.iloc[val])[:, 1]
    validation = metrics(y.iloc[val], scores, plan["threshold"])
    training = metrics(
        y.iloc[train], pipeline.predict_proba(x.iloc[train])[:, 1], plan["threshold"]
    )
    warnings = [
        "Scores are not calibrated probabilities.",
        "Repeated validation-based refinement can overfit validation data; final test data is reserved.",
    ]
    if any(y.iloc[part].sum() < 20 for part in (train, val, test)):
        warnings.append("Some splits contain fewer than 20 positives; results may be unstable.")
    if dropped or removed_duplicates:
        warnings.append(
            f"Dropped {dropped} rows with missing target labels and {removed_duplicates} exact duplicate rows before splitting."
        )
    bundle = {
        "pipeline": pipeline,
        "plan": plan,
        "features": features,
        "numeric": numeric,
        "classes": classes,
        "negative_label": next(c for c in classes if c != plan["positive_label"]),
        "validation_metrics": validation,
        "warnings": warnings,
    }
    joblib.dump(bundle, OUT / "model.joblib")
    return {
        "model": plan["model"],
        "plan": plan,
        "features": features,
        "training_metrics": training,
        "validation_metrics": validation,
        "baseline": {
            "accuracy_always_negative": float(1 - y.iloc[val].mean()),
            "recall_always_negative": 0.0,
            "constant_score_average_precision": float(y.iloc[val].mean()),
        },
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "charts": charts(y.iloc[val], scores, plan["threshold"]),
        "warnings": warnings,
        "evaluation_partition": "validation",
        "test_metrics": "Withheld until final approval",
    }
