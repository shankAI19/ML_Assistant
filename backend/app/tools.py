import hashlib
import json

from langchain.tools import tool

from .schemas import TaskRequest
from .workspace import new_id


def make_tools(session, workspace, runner, max_repairs=1):
    @tool
    def execute_ml_task(request: TaskRequest) -> dict:
        """Execute generated Python after user approval. Tasks: profile, train, evaluate,
        predict, report. code must be exactly `from ml_runtime import execute` followed
        by `execute(<Python literal dict>)`. Dict keys: task, plan (full plan or None),
        narrative (string). It must match the request. No arbitrary imports/statements.
        Training uses development data; evaluate freezes the current model and tests it.
        Provide a plain-language explanation of the work and its tradeoffs for approval.
        """
        try:
            with session.lock:
                if session.closed:
                    raise ValueError("Session ended")
                if session.failed_runs >= 1 + max_repairs:
                    raise ValueError(
                        "Repair budget exhausted. Explain the error and wait for a new user request."
                    )
                if not session.dataset:
                    raise ValueError("Upload a labeled CSV first")
                if request.task != "profile" and session.profile is None:
                    raise ValueError("Inspect the dataset first")
                if request.task == "train" and session.finalized:
                    raise ValueError("Model finalized; start a new session to retrain")
                if request.task == "train" and session.plan:
                    fixed = ("target", "positive_label", "seed", "split", "split_column")
                    if any(
                        getattr(request.plan, key) != getattr(session.plan, key) for key in fixed
                    ):
                        raise ValueError(
                            "Target and split are fixed after first training. Start a new session to change them."
                        )
                if request.task in {"evaluate", "predict", "report"} and not session.latest_train:
                    raise ValueError("Train a model first")
                if request.task == "evaluate" and session.finalized:
                    raise ValueError(
                        "Final evaluation already completed; do not tune on test results"
                    )
                if request.task in {"predict", "report"} and not session.finalized:
                    raise ValueError("Ask the user to approve final evaluation first")
                if request.task == "predict" and not session.prediction_dataset:
                    raise ValueError("Ask the user to upload a prediction CSV first")
                if request.task != "train" and request.plan is not None:
                    raise ValueError("Only train accepts a plan; other tasks use the frozen model")
                task_id = new_id()
                root = workspace.path(session.id)
                inputs = {"input.csv": root / "inputs" / session.dataset}
                if request.task in {"evaluate", "predict", "report"}:
                    previous = root / "outputs" / session.latest_train
                    inputs["model.joblib"] = previous / "model.joblib"
                if request.task == "predict":
                    inputs["prediction.csv"] = root / "inputs" / session.prediction_dataset
                if request.task == "report":
                    evidence = [
                        {"task": r["task"], "result": r["result"], "receipt": r["receipt"]}
                        for r in session.runs
                        if r["task"] != "report"
                    ]
                    inputs["evidence.json"] = json.dumps(evidence, allow_nan=False).encode()
                code_path = root / "code" / f"{task_id}.py"
                code_path.write_text(request.code, encoding="utf-8")
            artifacts, receipt = runner.run(session.id, request.code, inputs)
            result = json.loads(artifacts.get("result.json", b"{}"))
            with session.lock:
                if session.closed:
                    raise ValueError("Session ended; outputs discarded")
                if (
                    workspace.used_bytes(session.id) + sum(map(len, artifacts.values()))
                    > runner.settings.session_storage_mb * 1024**2
                ):
                    raise ValueError("Temporary session storage quota exceeded")
                destination = root / "outputs" / task_id
                destination.mkdir()
                for filename, data in artifacts.items():
                    (destination / filename).write_bytes(data)
                receipt["input_sha256"] = {
                    name: hashlib.sha256(
                        value.read_bytes() if hasattr(value, "read_bytes") else value
                    ).hexdigest()
                    for name, value in inputs.items()
                }
                (destination / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                run = {
                    "id": task_id,
                    "task": request.task,
                    "explanation": request.explanation,
                    "result": result,
                    "receipt": receipt,
                    "artifacts": list(artifacts) + ["receipt.json"],
                    "code": request.code,
                }
                session.runs.append(run)
                if request.task == "profile":
                    session.profile = result
                elif request.task == "train":
                    session.plan, session.latest_train = request.plan, task_id
                elif request.task == "evaluate":
                    session.finalized = True
                # Figures/artifact bytes stay out of LLM context; only computed summaries return.
                return {
                    "status": "completed",
                    "task": request.task,
                    "task_id": task_id,
                    "result": result,
                    "sandbox_destroyed": True,
                }
        except Exception as exc:  # noqa: BLE001 - return task failures to the agent for explanation
            session.failed_runs += 1
            return {
                "status": "failed",
                "error": str(exc)[:4000],
                "instruction": "Explain the failure. A changed task needs fresh approval.",
            }

    return [execute_ml_task]
