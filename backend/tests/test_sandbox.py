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


class InputSocket:
    def __init__(self, output=b"", *, timeout=False):
        self.data = None
        self.output = output
        self.closed = False
        self.timeout = timeout

    def settimeout(self, seconds):
        self.timeout_seconds = seconds

    def sendall(self, data):
        if self.timeout:
            raise TimeoutError("Transfer stalled")
        self.data = data

    def recv(self, size):
        chunk, self.output = self.output[:size], self.output[size:]
        return chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize("wrapped", [False, True])
def test_runner_enforces_boundary_and_removes_container(settings, wrapped):
    from types import SimpleNamespace

    from app.sandbox import SandboxRunner

    captured = {}
    transfer = InputSocket()
    connection = SimpleNamespace(_sock=transfer, close=transfer.close) if wrapped else transfer
    container = SimpleNamespace(id="1234", short_id="1234", image=SimpleNamespace(id="sha256:test"))
    container.start = lambda: None

    def forbidden_archive_upload(*args):
        raise AssertionError("PUT /archive must not be used with read-only rootfs")

    container.put_archive = forbidden_archive_upload
    container.get_archive = lambda path: ([archive_bytes()], {})
    container.remove = lambda **kw: captured.update(removed=kw)

    def create(**kwargs):
        captured.update(kwargs)
        return container

    client = SimpleNamespace(
        containers=SimpleNamespace(create=create),
        api=SimpleNamespace(
            exec_create=lambda *args, **kwargs: {"Id": "stage" if kwargs.get("stdin") else "task"},
            exec_start=lambda *args, **kwargs: connection if kwargs.get("socket") else [b"done"],
            exec_inspect=lambda *args: {"ExitCode": 0},
        ),
    )
    data = b"amount,label\n10,1\n\x00\xff"
    artifacts, receipt = SandboxRunner(settings, client).run("a" * 32, "pass", {"input.csv": data})
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
    assert transfer.closed
    with tarfile.open(fileobj=io.BytesIO(transfer.data)) as staged:
        assert staged.extractfile("task.py").read() == b"pass"
        assert staged.extractfile("input.csv").read() == data


@pytest.mark.parametrize("timeout", [True, False])
def test_staging_failures_stop_execution_and_remove_container(settings, timeout):
    from types import SimpleNamespace

    from app.sandbox import SandboxRunner

    transfer = InputSocket(b"file transfer failed", timeout=timeout)
    removed, created = [], []
    container = SimpleNamespace(
        id="sandbox", start=lambda: None, remove=lambda **kwargs: removed.append(kwargs)
    )

    def create_exec(*args, **kwargs):
        created.append(kwargs)
        return {"Id": "stage"}

    client = SimpleNamespace(
        containers=SimpleNamespace(create=lambda **kwargs: container),
        api=SimpleNamespace(
            exec_create=create_exec,
            exec_start=lambda *args, **kwargs: transfer,
            exec_inspect=lambda *args: {"ExitCode": 1},
        ),
    )
    with pytest.raises(RuntimeError, match="input transfer"):
        SandboxRunner(settings, client).run("a" * 32, "pass", {})
    assert len(created) == 1
    assert created[0]["stdin"] is True
    assert created[0]["user"] == "10001:10001"
    assert transfer.closed
    assert removed == [{"force": True, "v": True}]


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
