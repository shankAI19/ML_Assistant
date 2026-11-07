"""Exercise packaging and the portable launcher without executing ML on the host."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from app.sandbox import read_artifacts


def test_package_captures_runtime_and_inputs_and_launcher_uses_its_data(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "sandbox" / "experiment_export.py"
    spec = importlib.util.spec_from_file_location("experiment_export", source)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    runtime = tmp_path / "installed runtime"
    runtime.mkdir()
    (runtime / "experiment_export.py").write_bytes(source.read_bytes())
    stub = """import json
import os
from pathlib import Path

def execute(payload):
    work = Path(os.environ["ML_ASSISTANT_WORKDIR"])
    (work / "out").mkdir(exist_ok=True)
    value = {"payload": payload, "input": (work / "input.csv").read_text()}
    (work / "out" / "result.json").write_text(json.dumps(value))
"""
    (runtime / "ml_runtime.py").write_text(stub, encoding="utf-8")
    (runtime / "ml_tasks").mkdir()
    (runtime / "ml_tasks" / "__init__.py").write_text("# captured runtime")
    monkeypatch.setattr(exporter, "__file__", str(runtime / "experiment_export.py"))
    monkeypatch.setattr(
        exporter,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "example"}, version="1.2.3")],
    )
    work = tmp_path / "sandbox"
    work.mkdir()
    (work / "out").mkdir()
    task = "from ml_runtime import execute\nexecute({'task': 'profile', 'plan': None})"
    (work / "task.py").write_text(task)
    (work / "input.csv").write_text("x,y\n1,0\n2,1\n")
    (work / "model.joblib").write_bytes(b"saved fitted model")
    (work / "out" / "result.json").write_text("original output is not an input")
    exporter.export_experiment(work)
    extracted = tmp_path / "download with spaces"
    with ZipFile(work / "out" / "experiment.zip") as archive:
        assert archive.read("task.py") == (work / "task.py").read_bytes()
        assert archive.read("ml_runtime.py") == (runtime / "ml_runtime.py").read_bytes()
        assert archive.read("requirements.txt") == b"example==1.2.3\n"
        assert archive.read("data/model.joblib") == b"saved fitted model"
        assert "data/out/result.json" not in archive.namelist()
        assert "ml_tasks/__init__.py" in archive.namelist()
        archive.extractall(extracted)
    subprocess.run([sys.executable, str(extracted / "run.py")], cwd=tmp_path, check=True)
    result = json.loads((extracted / "data" / "out" / "result.json").read_text())
    assert result["payload"]["task"] == "profile"
    assert result["input"] == "x,y\n1,0\n2,1\n"


def test_zip_artifacts_survive_sandbox_collection():
    import tarfile

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        member = tarfile.TarInfo("out/experiment.zip")
        member.size = 4
        bundle.addfile(member, io.BytesIO(b"PK00"))
    assert read_artifacts(archive.getvalue(), 1024) == {"experiment.zip": b"PK00"}
