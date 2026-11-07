import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .workspace import Workspace, new_id


@dataclass
class Session:
    id: str
    token: str
    experiment_id: str = field(default_factory=new_id)
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    lock: Any = field(default_factory=threading.RLock)
    closed: bool = False
    busy: bool = False
    status: str = "ready"
    error: str | None = None
    messages: list = field(default_factory=list)
    pending: dict | None = None
    task_id: str | None = None
    agent: Any = None
    checkpointer: Any = None
    dataset: str | None = None
    dataset_filename: str | None = None
    prediction_dataset: str | None = None
    prediction_filename: str | None = None
    profile: dict | None = None
    plan: Any = None
    latest_train: str | None = None
    finalized: bool = False
    runs: list = field(default_factory=list)
    invocations: int = 0
    failed_runs: int = 0
    approved_training: dict | None = None


class SessionStore:
    def __init__(self, settings: Settings, workspace: Workspace, runner):
        self.settings, self.workspace, self.runner = settings, workspace, runner
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        self.pending_cleanup: set[str] = set()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")

    def create(self) -> Session:
        with self.lock:
            if len(self.sessions) >= self.settings.max_sessions:
                raise RuntimeError("The demo is at capacity. Please try again later.")
            session = Session(id=new_id(), token=secrets.token_urlsafe(32))
            self.workspace.create(session.id)
            self.sessions[session.token] = session
            return session

    def get(self, token: str, *, touch=True) -> Session:
        with self.lock:
            session = self.sessions.get(token)
            if session is None or session.closed:
                raise KeyError("Session expired. Start a new session.")
            now = time.monotonic()
            expired = (
                now - session.touched > self.settings.idle_timeout
                or now - session.created > self.settings.max_lifetime
            )
            if not expired:
                if touch:
                    session.touched = now
                return session
        self.delete(token)
        raise KeyError("Session expired. Start a new session.")

    def reset_experiment(self, session: Session):
        """Reset active experiment state after a replacement CSV has been validated.

        The caller holds session.lock and has reserved the idle session for upload.
        Previous artifact files remain downloadable by their existing URLs until
        normal session cleanup; they are not evidence for the new experiment.
        """
        if session.checkpointer:
            try:
                session.checkpointer.delete_thread(session.id)
            except Exception:  # noqa: BLE001 - a fresh agent never reuses the old checkpointer
                logging.getLogger(__name__).warning(
                    "Previous experiment checkpoint cleanup failed."
                )
        session.experiment_id = new_id()
        session.agent = session.checkpointer = None
        session.profile = session.plan = None
        session.latest_train = session.prediction_dataset = session.prediction_filename = None
        session.finalized = False
        session.pending = session.approved_training = None
        session.error = session.task_id = None
        session.status = "ready"
        session.invocations = session.failed_runs = 0
        session.messages.clear()
        session.runs.clear()

    def delete(self, token: str):
        with self.lock:
            session = self.sessions.pop(token, None)
        if session is None:
            return True
        # Do not wait for an LLM request to finish before revoking the session.
        with session.lock:
            session.closed = True
            session.pending = None
        cleaned = True
        try:
            self.runner.clean(session.id)
        except Exception:  # noqa: BLE001 - continue file deletion even when the daemon is unavailable
            cleaned = False
            self.pending_cleanup.add(session.id)
            logging.getLogger(__name__).warning("Docker cleanup failed; orphan sweep will retry.")
        try:
            self.workspace.delete(session.id)
        except OSError:
            cleaned = False
            self.pending_cleanup.add(session.id)
            logging.getLogger(__name__).warning("Temporary file cleanup will be retried.")
        finally:
            session.messages.clear()
            if session.checkpointer:
                session.checkpointer.delete_thread(session.id)
            session.agent = session.checkpointer = None
            session.runs.clear()
            session.profile = session.plan = None
        return cleaned

    def expire(self):
        now = time.monotonic()
        with self.lock:
            expired = [
                token
                for token, s in self.sessions.items()
                if now - s.touched > self.settings.idle_timeout
                or now - s.created > self.settings.max_lifetime
            ]
        for token in expired:
            self.delete(token)
        for session_id in list(self.pending_cleanup):
            try:
                self.runner.clean(session_id)
                self.workspace.delete(session_id)
                self.pending_cleanup.discard(session_id)
            except Exception:  # noqa: BLE001 - retry cleanup on the next sweep
                logging.getLogger(__name__).warning("Pending session cleanup will be retried.")
        try:
            with self.lock:
                active_ids = {s.id for s in self.sessions.values()}
            self.runner.clean_orphans(active_ids)
        except Exception:  # noqa: BLE001 - daemon outages must not stop the session expiry loop
            logging.getLogger(__name__).warning("Orphan container cleanup unavailable.")

    def close(self):
        for token in list(self.sessions):
            self.delete(token)
        self.pool.shutdown(wait=False, cancel_futures=True)
