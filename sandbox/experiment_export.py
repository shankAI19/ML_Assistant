"""Package the executed runtime and inputs for a portable, offline replay."""

from importlib.metadata import distributions
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

RUNNER = '''"""Run this experiment with Python 3.12 after installing requirements.txt."""
import os
import runpy
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ["ML_ASSISTANT_WORKDIR"] = str(root / "data")
os.environ.setdefault("MPLBACKEND", "Agg")
if __name__ == "__main__":
    runpy.run_path(str(root / "task.py"), run_name="__main__")
'''

README = """# Standalone experiment

This package contains the exact task.py sent to the sandbox, its runtime source,
the inputs used by that task, and pinned versions of the installed dependencies.
No application server, inference API key, or Docker is needed to run it locally.
The package includes your task data and, for evaluation/prediction, the fitted model.

Use Python 3.12. Extract the ZIP, open a terminal in its directory, and run:

    python -m venv .venv

Activate the environment on Windows:

    .venv\\Scripts\\activate

Or on macOS/Linux:

    source .venv/bin/activate

Then install dependencies and run:

    python -m pip install -r requirements.txt
    python run.py

Results are written to data/out/. You can replace the input CSVs in data/ while
keeping their filenames. task.py preserves the original task and approved plan.
Training packages retrain the model; evaluation/prediction packages use the saved
model. Report packages rebuild the report from saved evidence without LLM calls.

Download the original execution receipt separately from the app's results pane.
Dependency installation needs internet access; replay itself does not. Matching
versions improve reproducibility but do not guarantee identical results across
operating systems/hardware. Some platforms need a native OpenMP runtime for
LightGBM/XGBoost. Local replay runs with your user's permissions, outside the app's
resource-limited sandbox. Only load model.joblib from a trusted source.
"""


def export_experiment(work: Path):
    runtime = Path(__file__).resolve().parent
    requirements = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name") and dist.metadata["Name"].lower() != "pip"
        },
        key=str.lower,
    )
    with ZipFile(work / "out" / "experiment.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("run.py", RUNNER)
        archive.writestr("README.txt", README)
        archive.writestr("requirements.txt", "\n".join(requirements) + "\n")
        archive.write(work / "task.py", "task.py")
        archive.write(runtime / "ml_runtime.py", "ml_runtime.py")
        archive.write(runtime / "experiment_export.py", "experiment_export.py")
        for source in sorted((runtime / "ml_tasks").glob("*.py")):
            archive.write(source, f"ml_tasks/{source.name}")
        for source in sorted(work.iterdir()):
            if source.is_file() and source.name != "task.py":
                archive.write(source, f"data/{source.name}")
