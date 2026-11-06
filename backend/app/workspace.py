import shutil
from pathlib import Path
from uuid import uuid4


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
            raise ValueError("Invalid session identifier")
        path = (self.root / session_id).resolve()
        if path.parent != self.root:
            raise ValueError("Invalid workspace path")
        return path

    def create(self, session_id: str):
        path = self.path(session_id)
        for name in ("inputs", "code", "outputs"):
            (path / name).mkdir(parents=True, exist_ok=True)

    def delete(self, session_id: str):
        path = self.path(session_id)
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    def used_bytes(self, session_id: str) -> int:
        return sum(
            p.stat().st_size
            for p in self.path(session_id).rglob("*")
            if p.is_file() and not p.is_symlink()
        )

    def artifact(self, session_id: str, task_id: str, filename: str) -> Path:
        if not task_id.isalnum() or Path(filename).name != filename or "\\" in filename:
            raise ValueError("Invalid artifact")
        path = (self.path(session_id) / "outputs" / task_id / filename).resolve()
        if (
            not path.is_relative_to(self.path(session_id))
            or not path.is_file()
            or path.is_symlink()
        ):
            raise FileNotFoundError("Artifact not found")
        return path


def new_id() -> str:
    return uuid4().hex
