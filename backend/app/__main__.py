import uvicorn

from .config import load_settings

settings = load_settings()
uvicorn.run("app.main:app", host=settings.host, port=settings.port, workers=1)
