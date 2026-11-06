# ML Assistant

A guided binary-classification workbench for the Vultr **Blast Radius Zero** challenge.
Upload a CSV, discuss the objective, review each proposed action, approve execution,
refine a model, and download predictions and a dataset-specific DOCX report.

## Architecture

```text
Browser: vanilla JavaScript + Vite + Plotly.js
                     | same-origin /api requests
Vultr VM: Caddy -> FastAPI -> LangChain create_agent / LangGraph
                              |                |
                   Vultr Serverless      Disposable Docker tasks
                   Inference             (ML, plots, DOCX)
```

The frontend and backend are independent projects. `uv` is used only in `backend/`.
Vite builds frontend assets; FastAPI serves `frontend/dist/` in production. In development,
Vite proxies `/api` to FastAPI. No React, Streamlit, database, external tracing, or retained
session history is used. All LLM calls, including report narrative, go through the configured
Vultr inference endpoint. The default model is `qwen3.8-flash-next`.

## Local setup

Requires Python 3.12+, uv, Node.js 22.20+ with npm, and a running Linux-container Docker engine.
On Windows start Docker Desktop and use `npm.cmd` if PowerShell blocks `npm.ps1`.

1. From `backend/`, run `uv sync --locked` and copy `back.env.template` to `back.env`.
   Add your **Vultr Serverless Inference API key** to `back.env`.
2. From `frontend/`, run `npm ci` and copy `front.env.template` to `front.env`.
3. From the **repository root**, build the task image:

   ```sh
   docker build -f sandbox/Dockerfile -t ml-assistant-sandbox:latest .
   ```

4. Start the backend from `backend/`:

   ```sh
   uv run python -m app
   ```

5. In another terminal, start the frontend from `frontend/`:

   ```sh
   npm run dev
   ```

Open `http://127.0.0.1:5173`. Use `127.0.0.1` consistently for the browser and proxy.
The health indicator reports missing inference configuration or sandbox availability.
The app never substitutes fake results or executes generated code on the host.

For a production-style local build, run `npm run build` in `frontend/`, then open
`http://127.0.0.1:8000`. Restart FastAPI after the first frontend build so asset routes mount.
The Vite scripts use the native JavaScript config loader, avoiding an extra config-bundling
step. Node.js 22.20+ is the supported development/build environment.

## Configuration

* `backend/app/config.py` is the only backend env reader. It loads `backend/back.env`;
  process environment values override file values. It validates limits with Pydantic.
* `frontend/vite.config.js` reads the custom `frontend/front.env` file and exposes only
  the explicit public allowlist through `src/config.js`. Frontend config changes require
  restarting Vite or rebuilding production assets.
* Backend credentials never enter the frontend or task containers. The Docker socket is
  accessible only to the trusted control plane. A rootless daemon can be selected with
  `SANDBOX_DOCKER_HOST=unix:///run/user/<uid>/docker.sock` in `back.env`.
* Frontend and backend env templates are not interchangeable. Actual `*.env` files, virtual
  environments, node_modules, generated builds, and `.runtime/` are Git-ignored.

## Guided workflow

1. Upload a training CSV and describe the prediction objective.
2. Approve profiling. Review feature types, missingness, duplicates, and target candidates.
3. Confirm the binary target and exact positive label. Discuss false alarms versus missed
   events, IDs, group structure, time ordering, and possible leakage.
4. Review and approve preprocessing, model, weighting, split strategy, and threshold.
5. Run a development experiment. Review computed validation metrics and interactive plots;
   request refinements through chat. Each execution needs fresh approval.
6. Approve finalization. The current fitted pipeline and threshold are frozen and evaluated
   on the held-out test partition once. No full-data refit or subsequent tuning is performed.
7. Optionally upload prediction data, approve prediction, and download its CSV.
8. Request and approve a DOCX report using computed evidence and an LLM-written explanation.
9. Download outputs, then choose **End & delete**.

Supported classifiers: logistic regression, XGBoost, LightGBM, random forest. Numeric
missingness is imputed, categories are imputed and one-hot encoded, and logistic inputs
are scaled. Learned preprocessing and class weighting use training rows only. Unknown
prediction categories are ignored by the fitted encoder. There is no AutoML search.

Default splits are 60/20/20 train/validation/test. Group/time splits are supported when
specified; group fractions are approximate. Target, positive label, seed, and split policy
cannot change after the first experiment, preserving the same holdout during refinement.
The runtime rejects partitions without enough examples of both classes. Small positive
counts are explicitly flagged. Chronological splits reject ties across boundaries.

Results include precision, recall, specificity, F1, accuracy, balanced accuracy, ROC-AUC,
average precision, alert rate, false positives per 1,000 observations, and confusion counts.
ROC, PR, confusion-matrix, and threshold-tradeoff figures are rendered interactively by
Plotly.js. Average precision uses scikit-learn's AP definition, not trapezoidal PR area.
Undefined values are unavailable. Model scores are not presented as calibrated probabilities.
Unlabeled prediction uploads produce no accuracy/F1 claims. Labeled uploads may be used
to compare their computed metrics to validation/test results; prevalence affects comparisons.

Demo bounds: 25 MiB uploads by default, 100,000 rows, 200 columns, at most 50 categorical
features, and 100 distinct training values per categorical feature. Exclude high-cardinality
IDs. Resource limits can reject datasets below these bounds if their runtime exceeds budget.

## Agent and execution boundary

`agent.py` uses LangChain's `create_agent`, human-in-the-loop middleware, and LangGraph's
in-memory checkpoints. Approvals are bound to a pending action ID and cannot be replayed.
The model generates actual Python that calls the reviewed sandbox runtime with the approved
literal task plan. AST validation rejects other statements and mismatched plans. This is a
deliberately constrained ML product, not a general Python terminal. Training/evaluation
implementations live in `backend/app/task_templates/` and are copied into the task image;
they are never imported for ML execution in the host app.

The sandbox runner enforces:

* Separate container, non-root user, read-only root filesystem, dropped capabilities,
  and no privilege escalation. Docker's default seccomp profile remains enabled.
* No networking, credentials, host bind mounts, or Docker socket inside the container.
* CPU, memory/swap, PID, bounded writable tmpfs, log, artifact, and wall-clock limits.
  A container-local lifetime guard also stops/removes tasks if the control plane crashes.
* One active sandbox at a time; bounded active sessions, interactions, and repair attempts.
* Validated artifact types/paths; no unsafe tar extraction or model loading in FastAPI.
* Container removal after success, failure, timeout, or cancellation.

Execution receipts include image ID, code/input/artifact hashes, exit status, duration,
limits, and cleanup state. The UI offers code, model, prediction, report, and receipt
downloads. Metrics come from reviewed runtime calculations, not generated narrative.

Docker shares the host kernel; these controls are not an absolute escape-proof guarantee.
For the public demo keep the VM dedicated, patch its runtime, and use synthetic data in
the recording. gVisor can strengthen isolation later but is not required by this setup.

## Temporary data lifecycle

Active-session files live only under `.runtime/sessions/<random-id>/`; cookies contain an
unpredictable session token, not data. Browser state is in memory only. Polling does not
extend idle expiry. Defaults: 30-minute inactivity, 2-hour maximum lifetime, 60-second sweep,
and 256 MiB per-session file quota.

End-session revokes access, terminates its containers, deletes files, and removes checkpoints.
If Docker/filesystem cleanup temporarily fails, the API reports pending cleanup and the
sweeper retries. Startup deletes abandoned workspaces and containers; no recovery is offered.
An abrupt VM stop may leave files until the next startup. Deletion means removal from the
application filesystem, not forensic disk erasure. No app backups, request-body logging,
Docker logs, or external LangSmith tracing retain user content. Provider-side handling of
LLM requests is governed separately by Vultr. Downloads already saved by the user remain theirs.

## Verification and demo data

From `backend/`:

```sh
uv run pytest -m 'not docker'
uv run ruff check app tests
uv run ruff format --check app tests
uv run pytest -m docker
```

From `frontend/`: `npm test` and `npm run build`.
The Docker suite tests all four model families, evaluation, unlabeled prediction, DOCX,
and infinite-loop termination. It skips when Docker/image is unavailable; a skipped test
does not constitute containment verification. A live Vultr-key smoke test remains necessary
to confirm the selected model's tool-call behavior.

Generate synthetic data from the root:

```sh
python examples/generate_demo.py
```

Upload `.runtime/demo/training.csv`. Suggested request: “Help me predict fraud. Fraud=1
is positive. Exclude transaction_id and account_id. Use timestamp for a chronological
split; missed fraud matters more than false alarms. Explain your plan first.”
Prediction CSVs with and without labels are generated alongside it.

## Deploy on Vultr

1. Provision one Ubuntu CPU VM. A 4-vCPU/8-GiB instance is a starting sizing suggestion,
   not a requirement or guaranteed capacity. Install Docker, Python/uv, Node/npm, and Caddy.
2. Clone the repository to `/opt/ml-assistant`, create a dedicated `mlassistant` service
   user, grant it ownership of the repository and access to the Docker daemon.
   Docker control is privileged; never expose its socket through an API or sandbox.
3. Run `uv sync --locked --no-dev` in `backend/`; create `back.env`, supply the inference
   key, and set `COOKIE_SECURE=true`. Run `npm ci` and `npm run build` in `frontend/`.
4. Build the sandbox from the root with the command above. Preserve its image ID for
   reproducibility; a later rebuild can resolve different sandbox package versions.
5. Install `deploy/ml-assistant.service` into systemd and enable/start it. Adjust its paths
   if the checkout differs. Run one worker; multiple workers need a different session design.
6. Point a public domain at the VM. Replace the example hostname in `deploy/Caddyfile`,
   install it as Caddy's configuration, and reload Caddy. Expose 80/443; keep 8000 private
   and restrict SSH. Align Caddy's body cap if changing the upload limit.
7. Verify the live approve/execute/refine flow and `/api/health`. Record the **3-second
   containment check** button terminating an infinite loop, then run a normal task to show
   the application remains usable. Demonstrate End & delete too.

Required Vultr usage: VM-hosted control plane/web app, all agent LLM calls through Vultr
Serverless Inference, isolated task containers on Vultr. No managed database, object storage,
GPU, or Vultr management API is required. Redeem the $200 coupon in Vultr Console Billing's
hackathon Gift Code form; it is not an application environment variable.

Submit the public repository, public demo URL, recorded containment demo, presentation,
and the architecture/use-case explanation here. See the [challenge brief](https://lablab.build/ai-hackathons/vultr-hackathon).
