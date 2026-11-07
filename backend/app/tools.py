import hashlib
import json

from langchain.tools import tool
from pydantic import ValidationError

from .schemas import TaskRequest
from .workspace import new_id


def make_tools(session, workspace, runner, max_repairs=1):
    def run_task(request: TaskRequest) -> dict:
        """Dispatch validated work to an isolated sandbox; training is gated by train_model."""
        try:
            with session.lock:
                if session.closed:
                    raise ValueError("Session ended")
                if session.failed_runs >= 1 + max_repairs:
                    raise ValueError(
                        "Repair budget exhausted. Explain the error and wait for a new user request."
                    )
                if not session.dataset and request.task != "report":
                    raise ValueError("Upload a labeled CSV first")
                if request.task not in {"profile", "report"} and session.profile is None:
                    raise ValueError("Inspect the dataset first")
                if request.task == "train" and session.finalized:
                    raise ValueError(
                        "Model finalized; upload a training CSV to start a new experiment"
                    )
                if request.task == "train" and session.plan:
                    fixed = (
                        "target",
                        "positive_label",
                        "seed",
                        "split",
                        "split_column",
                        "validation_fraction",
                        "test_fraction",
                    )
                    if any(
                        getattr(request.plan, key) != getattr(session.plan, key) for key in fixed
                    ):
                        raise ValueError(
                            "Target and split are fixed within an experiment. Upload a training CSV to start a new experiment and change them."
                        )
                if request.task in {"evaluate", "predict"} and not session.latest_train:
                    raise ValueError("Train a model first")
                if request.task == "evaluate" and session.finalized:
                    raise ValueError(
                        "Final evaluation already completed; do not tune on test results"
                    )
                if request.task == "predict" and not session.finalized:
                    raise ValueError("Finalize the model first when the user is satisfied")
                if request.task == "predict" and not session.prediction_dataset:
                    raise ValueError("Ask the user to upload a prediction CSV first")
                if request.task != "train" and request.plan is not None:
                    raise ValueError("Only train accepts a plan; other tasks use the frozen model")
                task_id = new_id()
                root = workspace.path(session.id)
                inputs = (
                    {}
                    if request.task == "report"
                    else {"input.csv": root / "inputs" / session.dataset}
                )
                if request.task in {"evaluate", "predict"}:
                    previous = root / "outputs" / session.latest_train
                    inputs["model.joblib"] = previous / "model.joblib"
                if request.task == "predict":
                    inputs["prediction.csv"] = root / "inputs" / session.prediction_dataset
                if request.task == "report":
                    evidence = [
                        {
                            "sequence": index + 1,
                            "task": r["task"],
                            "result": r["result"],
                            "receipt": r["receipt"],
                        }
                        for index, r in enumerate(session.runs)
                        if r["task"] != "report"
                    ]
                    inputs["evidence.json"] = json.dumps(evidence, allow_nan=False).encode()
                    saved_charts = [
                        (index, run, filename)
                        for index, run in enumerate(session.runs)
                        for filename in run["result"].get("charts", [])
                    ]
                    for index, run, filename in saved_charts[-6:]:
                        inputs[f"chart_{index:03}_{run['task']}_{filename}"] = (
                            root / "outputs" / run["id"] / filename
                        )
                code_path = root / "code" / f"{task_id}.py"
                code = request.code
                code_path.write_text(code, encoding="utf-8")
            artifacts, receipt = runner.run(session.id, code, inputs)
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
                    "code": code,
                }
                session.runs.append(run)
                if request.task == "profile":
                    session.profile = result
                elif request.task == "train":
                    session.plan, session.latest_train = request.plan, task_id
                    session.approved_training = None
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
                "error": (
                    "The action could not complete because its settings or output failed validation. "
                    "Review the plan and try again."
                    if isinstance(exc, ValidationError)
                    else str(exc)[:4000]
                ),
                "instruction": "Explain the failure. Only a changed training plan needs fresh approval; an identical failed round may retry within the repair budget.",
            }

    @tool
    def execute_ml_task(request: TaskRequest) -> dict:
        """Run profile, evaluate, or predict without an extra approval. Never use for training.
        Submit task, explanation and optional plan/narrative. Python is generated by the backend.
        Only train accepts a plan. Prefer generate_report for reports.
        """
        if request.task == "train":
            return {"status": "failed", "error": "Use train_model: training requires approval"}
        return run_task(request)

    @tool
    def train_model(request: TaskRequest) -> dict:
        """Propose one training round for user approval after discussing the complete plan.
        request.task must be train. Include model, target, features, preprocessing, split,
        threshold and focus_metrics. The backend generates the executable Python wrapper.
        """
        if request.task != "train":
            return {"status": "failed", "error": "train_model only accepts train"}
        if request.payload() != session.approved_training:
            return {"status": "failed", "error": "This training plan has not been approved"}
        return run_task(request)

    @tool
    def generate_report(narrative: str = "") -> dict:
        """Generate a Markdown and DOCX progress report from existing results at ANY stage.
        Supply clear scientific English about the objective, observed findings,
        practical implications, and limitations. No JSON or configuration dumps.
        The runtime supplies methods, tables, and captioned figures from saved results.
        No plan, model, training, finalization, or code arguments are required.
        """
        payload = {"task": "report", "plan": None, "narrative": narrative[:20000]}
        return run_task(
            TaskRequest(
                **payload,
                explanation="Summarize the work completed so far in a downloadable report.",
            )
        )

    return [execute_ml_task, train_model, generate_report]
