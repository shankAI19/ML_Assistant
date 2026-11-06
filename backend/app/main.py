import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import REPO_ROOT, load_settings
from .sandbox import SandboxRunner
from .sessions import SessionStore
from .workflow import Workflow
from .workspace import Workspace


def create_app(settings=None, runner=None):
    settings = settings or load_settings()
    workspace = Workspace(settings.session_temp_root)
    runner = runner or SandboxRunner(settings)
    store = SessionStore(settings, workspace, runner)

    @asynccontextmanager
    async def lifespan(app):
        # No cross-restart recovery: delete abandoned session workspaces at startup.
        for entry in workspace.root.iterdir():
            if len(entry.name) == 32 and all(c in "0123456789abcdef" for c in entry.name):
                workspace.delete(entry.name)
        try:
            await asyncio.to_thread(runner.clean)
        except Exception:  # noqa: BLE001 - fail closed on execution, keep configuration UI available
            logging.getLogger(__name__).warning(
                "Docker cleanup unavailable at startup; execution requires Docker."
            )

        async def reap():
            while True:
                await asyncio.sleep(settings.cleanup_interval)
                await asyncio.to_thread(store.expire)

        reaper = asyncio.create_task(reap())
        try:
            yield
        finally:
            reaper.cancel()
            try:
                await reaper
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(store.close)

    app = FastAPI(
        title="ML Assistant",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.settings, app.state.workspace, app.state.runner = settings, workspace, runner
    app.state.store = store
    app.state.workflow = Workflow(settings, store, workspace, runner)

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
            length = request.headers.get("content-length")
            if length and (
                not length.isdigit() or int(length) > settings.max_upload_mb * 1024**2 + 65536
            ):
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(router)
    dist = REPO_ROOT / "frontend" / "dist"
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend():
        if (dist / "index.html").exists():
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
        return JSONResponse(
            {"detail": "Build frontend with npm run build, or use Vite on port 5173."}
        )

    return app


app = create_app()
