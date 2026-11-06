import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeRunner:
    def __init__(self):
        self.settings = Settings()
        self.calls = []
        self.cleaned = []

    def available(self):
        return True

    def clean(self, session_id=None):
        self.cleaned.append(session_id)

    def clean_orphans(self, active_ids):
        return None

    def run(self, session_id, code, inputs, **kwargs):
        self.calls.append((session_id, code, inputs))
        return {
            "result.json": json.dumps({"rows": 100, "columns": [], "duplicates": 0}).encode()
        }, {
            "elapsed_seconds": 0.1,
            "memory_mb": 256,
            "timeout_seconds": 120,
            "code_sha256": "test",
            "sandbox_destroyed": True,
        }


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def settings(tmp_path):
    return Settings(session_temp_root=tmp_path / "sessions")


@pytest.fixture
def client(settings, runner):
    app = create_app(settings, runner)
    with TestClient(app) as client:
        yield client


def wait_ready(session):
    deadline = time.monotonic() + 10
    while session.busy and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not session.busy, "Agent did not finish"
