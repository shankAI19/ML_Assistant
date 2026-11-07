from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from .schemas import ApprovalRequest, ChatRequest
from .workspace import new_id

router = APIRouter(prefix="/api")


def current(request: Request, *, touch=True):
    try:
        return request.app.state.store.get(request.cookies.get("ml_session", ""), touch=touch)
    except KeyError as exc:
        raise HTTPException(401, str(exc)) from None


def snapshot(session):
    with session.lock:
        return {
            "id": session.id,
            "experiment_id": session.experiment_id,
            "status": session.status,
            "busy": session.busy,
            "error": session.error,
            "messages": list(session.messages),
            "pending": session.pending,
            "task_id": session.task_id,
            "dataset_uploaded": bool(session.dataset),
            "training_filename": session.dataset_filename,
            "prediction_uploaded": bool(session.prediction_dataset),
            "prediction_filename": session.prediction_filename,
            "finalized": session.finalized,
            "runs": list(session.runs),
        }


@router.get("/health")
def health(request: Request):
    settings = request.app.state.settings
    return {
        "status": "ok",
        "llm_configured": bool(settings.api_key.get_secret_value()),
        "sandbox_ready": request.app.state.runner.available(),
        "max_upload_mb": settings.max_upload_mb,
    }


@router.post("/sessions")
def create_session(request: Request, response: Response):
    store = request.app.state.store
    existing = request.cookies.get("ml_session")
    if existing:
        try:
            return snapshot(store.get(existing))
        except KeyError:
            pass
    try:
        session = store.create()
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from None
    response.set_cookie(
        "ml_session",
        session.token,
        httponly=True,
        samesite="strict",
        secure=request.app.state.settings.secure_cookie,
        path="/api",
    )
    return snapshot(session)


@router.get("/sessions/current")
def get_session(request: Request):
    # Polls do not extend inactivity retention.
    return snapshot(current(request, touch=False))


@router.delete("/sessions/current")
def delete_session(request: Request, response: Response):
    cleaned = request.app.state.store.delete(request.cookies.get("ml_session", ""))
    response.delete_cookie("ml_session", path="/api")
    return {"deleted": cleaned, "cleanup_pending": not cleaned}


@router.post("/uploads")
async def upload(
    request: Request,
    kind: Literal["training", "prediction"],
    file: Annotated[UploadFile, File()],
    replace_training: bool = False,
):
    session = current(request)
    if Path(file.filename or "").suffix.lower() != ".csv":
        await file.close()
        raise HTTPException(400, "Upload a CSV file")
    with session.lock:
        if session.busy:
            raise HTTPException(409, "A task is running. Upload again when it finishes.")
        if kind == "training" and session.dataset and not replace_training:
            raise HTTPException(
                409, "Confirm this upload as training data to start a new experiment."
            )
        session.busy = True
    path = request.app.state.workspace.path(session.id) / "inputs" / f"{kind}.csv"
    staging = path.with_name(f".{kind}-{new_id()}.upload")
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with staging.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "CSV exceeds the configured upload limit")
                if (
                    request.app.state.workspace.used_bytes(session.id) + len(chunk)
                    > request.app.state.settings.session_storage_mb * 1024**2
                ):
                    raise HTTPException(413, "Temporary session storage quota exceeded")
                destination.write(chunk)
        if not size:
            raise HTTPException(400, "The CSV is empty")
        with session.lock:
            if session.closed:
                raise HTTPException(401, "Session ended")
            staging.replace(path)
            if kind == "training":
                if session.dataset:
                    request.app.state.store.reset_experiment(session)
                session.dataset = path.name
                session.dataset_filename = (file.filename or "training.csv")[:200]
            else:
                session.prediction_dataset = path.name
                session.prediction_filename = (file.filename or "prediction.csv")[:200]
            session.messages.append(
                {"role": "system", "content": f"{kind.title()} CSV uploaded ({size:,} bytes)."}
            )
        return {"uploaded": True, "kind": kind, "bytes": size}
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    finally:
        with session.lock:
            session.busy = False
        await file.close()


@router.post("/chat")
def chat(body: ChatRequest, request: Request):
    try:
        return {
            "task_id": request.app.state.workflow.submit(current(request), message=body.message)
        }
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/approvals")
def approve(body: ApprovalRequest, request: Request):
    try:
        return {"task_id": request.app.state.workflow.submit(current(request), approval=body)}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/artifacts/{task_id}/{filename}")
def artifact(task_id: str, filename: str, request: Request):
    session = current(request)
    try:
        path = request.app.state.workspace.artifact(session.id, task_id, filename)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Artifact not found") from None
    return FileResponse(path, filename=path.name, headers={"Cache-Control": "no-store"})


@router.get("/code/{task_id}")
def code(task_id: str, request: Request):
    session = current(request)
    with session.lock:
        run = next((r for r in session.runs if r["id"] == task_id), None)
        if run is None:
            raise HTTPException(404, "Code not found")
        return Response(
            run["code"],
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{task_id}.py"'},
        )


@router.post("/containment-demo")
def containment_demo(request: Request):
    session = current(request)
    with session.lock:
        if session.busy or session.pending:
            raise HTTPException(409, "Finish the current action first")
        session.busy, session.status = True, "working"

    # This explicit button approves a fixed infinite loop, never arbitrary host code.
    def run_demo():
        try:
            request.app.state.runner.run(session.id, "while True:\n    pass\n", {}, timeout=3)
            text = "Containment check unexpectedly completed; investigate the runner."
        except Exception as exc:  # noqa: BLE001 - isolate failures of an explicitly requested demo
            if "exceeded its 3s time limit" in str(exc):
                text = "Containment check passed: infinite loop killed at 3 seconds; container removed."
            else:
                text = "Containment check could not run. Verify Docker and build the sandbox image."
        with session.lock:
            if not session.closed:
                session.messages.append({"role": "system", "content": text})
                session.busy, session.status = False, "ready"

    request.app.state.store.pool.submit(run_demo)
    return {"started": True}
