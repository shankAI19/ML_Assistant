"""Fixed sandbox entry point imported by the agent-generated, approved task script."""

from experiment_export import export_experiment
from ml_tasks import evaluate, predict, profile, report, train
from ml_tasks.common import save_json
from ml_tasks.paths import OUT, WORK


def execute(payload):
    OUT.mkdir(exist_ok=True)
    handlers = {
        "profile": profile.run,
        "train": train.run,
        "evaluate": evaluate.run,
        "predict": predict.run,
        "report": report.run,
    }
    result = handlers[payload["task"]](payload)
    save_json("result.json", result)
    export_experiment(WORK)
    print(f"Completed {payload['task']}. Results written to {OUT}.")
