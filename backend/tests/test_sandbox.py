import io
import tarfile

import pytest

from app.sandbox import read_artifacts


def archive_bytes():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        entry = tarfile.TarInfo("out/result.json")
        entry.size = 2
        archive.addfile(entry, io.BytesIO(b"{}"))
    return data.getvalue()


def test_runner_enforces_boundary_and_removes_container(settings):
    from types import SimpleNamespace

    from app.sandbox import SandboxRunner

    captured = {}
    container = SimpleNamespace(id="1234", short_id="1234", image=SimpleNamespace(id="sha256:test"))
    container.start = lambda: None
    container.put_archive = lambda path, data: None
    container.get_archive = lambda path: ([archive_bytes()], {})
    container.remove = lambda **kw: captured.update(removed=kw)

    def create(**kwargs):
        captured.update(kwargs)
        return container

    client = SimpleNamespace(
        containers=SimpleNamespace(create=create),
        api=SimpleNamespace(
            exec_create=lambda *args, **kwargs: {"Id": "exec"},
            exec_start=lambda *args, **kwargs: [b"done"],
            exec_inspect=lambda *args: {"ExitCode": 0},
        ),
    )
    artifacts, receipt = SandboxRunner(settings, client).run("a" * 32, "pass", {})
    assert artifacts == {"result.json": b"{}"}
    assert captured["network_mode"] == "none"
    assert captured["read_only"] is True
    assert captured["user"] == "10001:10001"
    assert captured["cap_drop"] == ["ALL"]
    assert captured["mem_limit"] == "2048m"
    assert "volumes" not in captured and "mounts" not in captured
    assert set(captured["environment"]) == {
        "HOME",
        "MPLCONFIGDIR",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
    }
    assert captured["removed"] == {"force": True, "v": True}
    assert receipt["sandbox_destroyed"] is True


@pytest.mark.parametrize(
    "name,kind",
    [("out/../../escape.json", "file"), ("out/link.json", "link"), ("out/evil.html", "file")],
)
def test_unsafe_artifacts_rejected(name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        entry = tarfile.TarInfo(name)
        if kind == "link":
            entry.type, entry.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(entry)
        else:
            entry.size = 2
            archive.addfile(entry, io.BytesIO(b"{}"))
    with pytest.raises(ValueError):
        read_artifacts(data.getvalue(), 1000)
