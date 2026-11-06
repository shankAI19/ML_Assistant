# Backend

Run from this directory:

```sh
uv sync --locked
cp back.env.template back.env
# Set VULTR_INFERENCE_API_KEY in back.env.
uv run python -m app
```

For hot reload: `uv run uvicorn app.main:app --reload --no-access-log`.
Use one worker: sessions and LangGraph checkpoints are in memory, intentionally without
cross-process persistence. The application does not start or stop a Docker daemon.

`app/config.py` is the sole environment reader. Relative workspace paths resolve from
the repository root. `back.env` never enters the frontend or sandbox.

Dependencies belong in `pyproject.toml`; `uv.lock` records their resolution.
`requirements.txt` is an export for tools that need pip input. Regenerate it with:

```sh
uv export --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

Checks:

```sh
uv run pytest -m 'not docker'
uv run ruff check app tests
uv run ruff format --check app tests
uv run pytest -m docker
```

Docker tests require the image built using the command in the root README. They run
ML code only inside Docker, including the training/prediction/report tests. They skip
when the daemon/image is unavailable. Unit tests use a scripted test-only model and a
fake runner to verify the real LangGraph approval/resume behavior without an API key.
There is no mock execution mode in the product.
