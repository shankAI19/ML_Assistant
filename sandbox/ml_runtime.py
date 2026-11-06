"""Fixed sandbox entry point imported by the agent-generated, approved task script."""
from ml_tasks import evaluate, predict, profile, report, train
from ml_tasks.common import OUT, save_json


def execute(payload):
    OUT.mkdir(exist_ok=True)
    handlers = {"profile": profile.run, "train": train.run, "evaluate": evaluate.run,
                "predict": predict.run, "report": report.run}
    result = handlers[payload["task"]](payload)
    save_json("result.json", result)
    print(f"Completed {payload['task']}. Results written to /work/out.")
