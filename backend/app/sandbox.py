"""The sole Docker boundary: no host execution fallback, bind mounts, or inherited env."""

import hashlib
import io
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path, PurePosixPath

import docker

from .config import Settings

LABEL = "ml-assistant.session"


def read_artifacts(archive: bytes, limit: int) -> dict[str, bytes]:
    result = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        total = 0
        for member in bundle:
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or len(name.parts) != 2
                or name.parts[0] != "out"
                or name.name.startswith(".")
                or "\\" in name.name
                or ":" in name.name
                or any(ord(c) < 32 for c in name.name)
                or member.size < 0
            ):
                raise ValueError("Unsafe artifact path or file type")
            if name.suffix not in {".json", ".csv", ".png", ".docx", ".joblib", ".txt"}:
                raise ValueError("Unsupported artifact type")
            total += member.size
            if total > limit or name.name in result or len(result) >= 30:
                raise ValueError("Artifact limit exceeded")
            stream = bundle.extractfile(member)
            result[name.name] = stream.read(limit + 1)
    return result


class SandboxRunner:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client
        self.slot = threading.BoundedSemaphore(1)

    @property
    def client(self):
        if self._client is None:
            self._client = docker.DockerClient(base_url=self.settings.docker_host, timeout=10)
        return self._client

    def available(self) -> bool:
        try:
            self.client.ping()
            self.client.images.get(self.settings.sandbox_image)
            return True
        except Exception:  # noqa: BLE001 - health must report unavailable instead of exposing SDK errors
            return False

    def clean(self, session_id: str | None = None):
        query = LABEL if session_id is None else f"{LABEL}={session_id}"
        for container in self.client.containers.list(all=True, filters={"label": query}):
            container.remove(force=True, v=True)

    def clean_orphans(self, active_ids):
        for container in self.client.containers.list(all=True, filters={"label": LABEL}):
            if container.labels.get(LABEL) not in active_ids:
                container.remove(force=True, v=True)

    def run(self, session_id: str, code: str, inputs: dict[str, Path | bytes], *, timeout=None):
        if not self.slot.acquire(blocking=False):
            raise RuntimeError("Another sandbox is running. Please try again shortly.")
        container = None
        started = time.monotonic()
        limit = self.settings.max_output_mb * 1024 * 1024
        deadline = timeout or self.settings.sandbox_timeout
        try:
            # The image includes ml_runtime; all user files live in a bounded tmpfs.
            container = self.client.containers.create(
                image=self.settings.sandbox_image,
                # Container-local lifetime guard survives a control-plane crash.
                command=["timeout", "--signal=KILL", f"{deadline + 30}s", "sleep", "infinity"],
                auto_remove=True,
                user="10001:10001",
                working_dir="/work",
                network_mode="none",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit=f"{self.settings.sandbox_memory_mb}m",
                memswap_limit=f"{self.settings.sandbox_memory_mb}m",
                nano_cpus=int(self.settings.sandbox_cpus * 1_000_000_000),
                pids_limit=self.settings.sandbox_pids,
                init=True,
                tmpfs={
                    "/work": f"rw,nosuid,nodev,size={self.settings.sandbox_tmpfs_mb}m,uid=10001,gid=10001,mode=0700",
                    "/tmp": "rw,noexec,nosuid,nodev,size=32m,mode=1777",
                },
                environment={
                    "HOME": "/tmp",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                labels={LABEL: session_id},
                log_config=docker.types.LogConfig(type="none"),
            )
            container.start()
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as bundle:
                for name, value in {"task.py": code.encode(), **inputs}.items():
                    if PurePosixPath(name).name != name or "\\" in name:
                        raise ValueError("Invalid sandbox input name")
                    data = value.read_bytes() if isinstance(value, Path) else value
                    info = tarfile.TarInfo(name)
                    info.size, info.uid, info.gid, info.mode = len(data), 10001, 10001, 0o400
                    bundle.addfile(info, io.BytesIO(data))
            container.put_archive("/work", buffer.getvalue())
            exec_id = self.client.api.exec_create(
                container.id, ["python", "-I", "/work/task.py"], user="10001:10001", workdir="/work"
            )["Id"]

            def execute():
                chunks, size = [], 0
                for chunk in self.client.api.exec_start(exec_id, stream=True):
                    size += len(chunk)
                    if size > 64 * 1024:
                        raise RuntimeError("Sandbox log limit exceeded")
                    chunks.append(chunk)
                status = self.client.api.exec_inspect(exec_id)
                return status["ExitCode"], b"".join(chunks).decode(errors="replace")

            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(execute)
            try:
                exit_code, log = future.result(timeout=deadline)
            except TimeoutError:
                raise RuntimeError(f"Sandbox exceeded its {deadline}s time limit") from None
            finally:
                # A timeout is terminated by removing the entire container below.
                pool.shutdown(wait=False, cancel_futures=True)
            if exit_code != 0:
                raise RuntimeError(f"Sandbox exited with code {exit_code}: {log[-4000:]}")
            chunks, _ = container.get_archive("/work/out")
            archive = bytearray()
            for chunk in chunks:
                archive.extend(chunk)
                if len(archive) > limit + 1024 * 1024:
                    raise RuntimeError("Sandbox output limit exceeded")
            artifacts = read_artifacts(bytes(archive), limit)
            return artifacts, {
                "container_id": container.short_id,
                "network": "none",
                "memory_mb": self.settings.sandbox_memory_mb,
                "timeout_seconds": deadline,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "image_id": container.image.id,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "artifact_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in artifacts.items()},
                "exit_code": exit_code,
                "log": log[-4000:],
                "sandbox_destroyed": True,
            }
        finally:
            try:
                if container is not None:
                    try:
                        container.remove(force=True, v=True)
                    except docker.errors.NotFound:
                        # The independent container lifetime guard may have removed it.
                        pass
            finally:
                self.slot.release()
