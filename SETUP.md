# Deploy ML Assistant on Vultr

This guide deploys the complete application on one Vultr VM: production frontend, FastAPI backend, Docker execution sandboxes, and a public HTTPS endpoint. All agent LLM requests go through Vultr Serverless Inference.

The [Blast Radius Zero challenge](https://lablab.build/ai-hackathons/vultr-hackathon) calls for a VM-hosted application with isolated execution, a public demo URL, repository documentation, and a recorded demonstration containing a containment moment. An SSH tunnel is useful for troubleshooting but is not the final public deployment.

## Before you start

Have the following ready:

- Access to Vultr Console and your hackathon credits, if applicable.
- The GitHub repository URL containing this code.
- A Vultr Serverless Inference API key and access to a tool-calling model.
- A domain or subdomain whose DNS you can edit for public HTTPS access.
- An SSH client on your computer. Windows PowerShell includes one on most current installations.

The examples use **Ubuntu 24.04 LTS x64**, an SSH user named `linuxuser`, and the checkout directory `~/ML_Assistant`. Replace `YOUR_VM_IP`, `YOUR_REPOSITORY_URL`, and `ml.your-domain.com` with your values. If your login username differs, use it instead of `linuxuser`.

Unless a block is labeled **local PowerShell**, run its commands in the VM's SSH terminal. Work through the numbered steps in order. If a command fails, resolve that failure before continuing.

## 1. Apply credits and prepare SSH access

If you received a hackathon gift code, apply it through the hackathon promotion page in Vultr Console's Billing area and confirm the resulting credit balance. The code is not an application setting and does not belong in either environment file. The event's [credit instructions](https://lablab.build/ai-hackathons/vultr-hackathon) link to the promotion page.

If you already have an SSH key you intend to use, reuse it. Otherwise generate one on your computer.

**Local PowerShell:**

```powershell
ssh-keygen -t ed25519 -C "ml-assistant" -f "$env:USERPROFILE\.ssh\vultr_ml_assistant"
Get-Content "$env:USERPROFILE\.ssh\vultr_ml_assistant.pub"
```

`$env:USERPROFILE` means your Windows user directory. Keep the private key on your computer. Add the **entire public-key line** from the `.pub` file to Vultr's SSH Keys area, available under Orchestration in the console used for this project. Do not replace individual pieces of the example text in the console form.

## 2. Create the VM

In Vultr Console, open **Compute → Instances → Create Instance**.

1. Select a Shared CPU VM. A configuration with 4 vCPUs and 8 GB RAM is a practical starting point for this demo, not a guaranteed capacity requirement. No GPU is needed.
2. Choose an available location.
3. Select **Ubuntu 24.04 LTS x64** under Operating System.
4. Select your SSH key and enable Limited User Login if available. Note the actual username shown after deployment.
5. Give the instance a hostname and label, such as `ml-assistant`.
6. Use a public IP. Disable automatic backups for this deployment so session files are not included in scheduled VM backups; avoid manual snapshots containing active session data as well.
7. Deploy and wait until the instance is running. Copy its public IP.

For a Vultr firewall group, allow inbound TCP 22 from your administrative IP and TCP 80/443 for public web access. Apply equivalent IPv6 rules if publishing an IPv6 address. Port 8000 and Docker's daemon must not be exposed publicly.

## 3. Connect and update Ubuntu

**Local PowerShell:**

```powershell
ssh -i "$env:USERPROFILE\.ssh\vultr_ml_assistant" linuxuser@YOUR_VM_IP
```

Verify the host fingerprint through the instance console before accepting the first connection. Enter your private-key passphrase if you set one.

**On the VM:**

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git curl ca-certificates python3-venv gnupg ufw
```

If Ubuntu requests a reboot:

```bash
sudo reboot
```

Wait briefly, then reconnect using the same SSH command.

## 4. Install Docker Engine on the VM

These commands follow the [official Docker Ubuntu installation instructions](https://docs.docker.com/engine/install/ubuntu/) for a fresh Ubuntu 24.04 VM. Docker Desktop is not needed on your laptop.

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Disconnect with `exit`, then reconnect so the group membership takes effect. Verify:

```bash
docker run --rm hello-world
docker ps
```

Both commands should work without `sudo`. Docker group membership grants powerful access to the VM; only the trusted application operator should have it. The application never passes that access into task containers.

## 5. Install uv and Node.js

Install uv using its [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/ml-assistant-install-uv.sh
sh /tmp/ml-assistant-install-uv.sh
source "$HOME/.local/bin/env"
uv --version
```

Install Node.js 22 using [NodeSource's distribution repository](https://github.com/nodesource/distributions):

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/ml-assistant-setup-node.sh
sudo -E bash /tmp/ml-assistant-setup-node.sh
sudo apt install -y nodejs
node --version
npm --version
```

Use Node.js 22.20 or newer within the 22.x series. Python dependencies are managed with uv in `backend/`; JavaScript dependencies are managed with npm in `frontend/`.

## 6. Clone the repository

```bash
cd ~
git clone YOUR_REPOSITORY_URL ML_Assistant
cd ~/ML_Assistant
```

Cloning into `/home/linuxuser/ML_Assistant` is fine. Do not run `uv init` at the repository root; the Python project already exists under `backend/`.

A public repository can be cloned using its HTTPS URL without GitHub authentication. For a private repository, use an SSH key or GitHub CLI authentication rather than your Google password. One option is `sudo apt install -y gh`, followed by `gh auth login --hostname github.com --git-protocol https --web`, before cloning.

## 7. Configure Vultr Serverless Inference

In Vultr Console, open **Serverless Inference**, select or create your inference subscription, and copy its actual API key from the subscription details. The subscription label is not the key. This application needs the inference key, not a Vultr VM management API key. See [Vultr's inference authentication documentation](https://api.vultrinference.com/docs/get-started/authentication).

Create the backend configuration:

```bash
cd ~/ML_Assistant/backend
uv sync --locked
cp -n back.env.template back.env
chmod 600 back.env
nano back.env
```

Set these values in the file, keeping the remaining defaults initially:

```dotenv
VULTR_INFERENCE_API_KEY=YOUR_ACTUAL_INFERENCE_KEY
VULTR_INFERENCE_BASE_URL=https://api.vultrinference.com/v1
VULTR_INFERENCE_MODEL=qwen3.8-flash-next
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.2
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
COOKIE_SECURE=true
```

Save in nano with **Ctrl+O**, press **Enter**, then exit with **Ctrl+X**. Choose an available tool-calling model from the [Vultr model catalog](https://api.vultrinference.com/docs/models) if changing the default.

`app/config.py` centrally reads `back.env`; existing process environment values take precedence. Never commit the real key or put it in frontend settings, Docker build arguments, or sandbox environment variables. Neither the coupon nor VM login credentials belongs in this file.

Check that an authenticated LLM request succeeds without displaying the key:

```bash
uv run python - <<'PY'
from app.config import load_settings
from app.llm import build_llm

response = build_llm(load_settings()).invoke("Reply with only OK.")
print(response.content)
PY
```

A successful model response confirms the request. The public model catalog and the application's `llm_configured` health flag do not establish that the key is valid or that the selected model can execute the agent workflow.

## 8. Build the sandbox image

Run this from the repository root. The final dot is required: it supplies the Docker build context.

```bash
cd ~/ML_Assistant
docker build -f sandbox/Dockerfile -t ml-assistant-sandbox:latest .
docker image inspect ml-assistant-sandbox:latest --format '{{.Id}}'
```

The image contains the ML libraries and reviewed task runtime. Each task gets a separate non-root, network-disabled container with a read-only root filesystem, bounded writable storage, CPU/memory/PID limits, and a timeout. Inputs and outputs are transferred without exposing host mounts, the Docker socket, or credentials. Containers are removed after tasks.

Active-session files needed between tasks remain under `.runtime/sessions/` until the session ends or expires. Do not replace this with persistent storage or enable session backups for the demo.

## 9. Build the production frontend

```bash
cd ~/ML_Assistant/frontend
cp -n front.env.template front.env
nano front.env
```

Keep the API on the same origin:

```dotenv
FRONTEND_APP_TITLE="ML Assistant"
FRONTEND_API_BASE_URL=/api
FRONTEND_MAX_UPLOAD_MB=25
FRONTEND_DEV_API_TARGET=http://127.0.0.1:8000
```

The development proxy target is not used by the production build. Vite reads `front.env` and exposes an explicit allowlist of public configuration; browser JavaScript does not read the file at runtime.

Build and check the frontend:

```bash
npm ci
npm test
npm run build
```

This creates `frontend/dist/`, which FastAPI serves. Do not run `npm run dev` or expose port 5173 as the production website.

## 10. Run backend and real Docker checks

```bash
cd ~/ML_Assistant/backend
uv run ruff check app tests
uv run ruff format --check app tests
uv run pytest -m 'not docker' -q
uv run pytest -m docker -q
```

The Docker suite exercises actual sandbox training, evaluation, prediction, reports, configurable splits, feature expansion, and timeout cleanup. Skipped Docker tests do not verify execution or containment. Resolve failures before relying on the public demo.

These checks do not replace a browser test with the actual configured LLM, which comes after public deployment.

## 11. Run the backend as a persistent service

Stop any manually started backend with **Ctrl+C** before starting the service. The supplied service template assumes `/opt/ml-assistant` and a user named `mlassistant`; the command below adapts it to your current SSH user and `~/ML_Assistant` checkout.

```bash
cd ~/ML_Assistant
sed -e "s/^User=mlassistant$/User=$USER/" \
    -e "s/^Group=mlassistant$/Group=$(id -gn)/" \
    -e "s|/opt/ml-assistant|$HOME/ML_Assistant|g" \
    deploy/ml-assistant.service | sudo tee /etc/systemd/system/ml-assistant.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ml-assistant
sudo systemctl status ml-assistant --no-pager
curl -fsS http://127.0.0.1:8000/api/health
```

The service should be active, and health should show `sandbox_ready: true` and `llm_configured: true`. The service binds to `127.0.0.1:8000`, uses one worker, and remains running when you close SSH.

Use **one Uvicorn worker** because sessions and agent checkpoints are in memory. The service's `ExecStart` fixes port 8000; changing `BACKEND_PORT` alone does not change this service. Keep the service and Caddy upstream ports aligned.

## 12. Point a domain at the VM

At your DNS provider, create an **A record** for your chosen hostname, such as `ml.your-domain.com`, pointing to `YOUR_VM_IP`. Only publish an AAAA record if IPv6 is configured and reachable on the VM.

Wait until the hostname resolves to the correct address. Both public TCP ports 80 and 443 must reach the VM for this HTTPS setup.

## 13. Install Caddy and enable public HTTPS

Install the stable package using [Caddy's official Ubuntu instructions](https://caddyserver.com/docs/install#debian-ubuntu-raspbian):

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg
sudo chmod o+r /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

Copy the repository's Caddyfile and replace `ml.example.com` with your real hostname:

```bash
sudo cp ~/ML_Assistant/deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile
```

Leave the upstream as `127.0.0.1:8000`. Caddy proxies both `/` and `/api` to FastAPI, so frontend assets and API calls use one public URL. If changing the upload size, align the backend limit, frontend limit, and Caddy's request-body cap.

Allow web traffic through the VM firewall, permitting SSH before enabling it:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

These commands assume the standard SSH port 22 on a fresh VM. Review any existing firewall rules; do not leave rules exposing port 8000 or Docker. A separate Vultr firewall must also allow 80/443.

Open **`https://ml.your-domain.com`** from your computer, without an SSH tunnel. Caddy manages the certificate once DNS and inbound connectivity are correct. `COOKIE_SECURE=true` is required for this deployment's session cookie configuration.

## 14. Verify the public workflow end to end

Generate the included synthetic data on the VM:

```bash
cd ~/ML_Assistant
python3 examples/generate_demo.py
```

Download the generated examples to your computer so you can upload them through the browser.

**Local PowerShell, from a directory where you want the examples saved:**

```powershell
scp -i "$env:USERPROFILE\.ssh\vultr_ml_assistant" "linuxuser@YOUR_VM_IP:~/ML_Assistant/.runtime/demo/*.csv" .
```

In the public application:

1. Upload `training.csv` and ask the assistant to inspect it.
2. Explain that `fraud` is the target and `1` is the positive class. Discuss excluding identifiers and how `timestamp` should be handled. For a simple initial baseline, exclude `transaction_id`, `account_id`, and `timestamp`, and use logistic regression with a stratified split.
3. Agree on the split, class weighting, threshold, and focus metrics. Approve the complete training round once.
4. Confirm that real validation metrics and interactive charts appear in Results. Ask a follow-up question without an approval prompt.
5. Request a report before finalization and download its Markdown, DOCX, and illustrated report ZIP. Verify that methods and metrics are readable tables, saved figures have captions, validation is distinguished from test performance, and the report contains no raw configuration or receipt JSON. Extract the ZIP to view Markdown with its figure images.
6. Request finalization. A reserved internal test set is assessed once; a plan without one finalizes without inventing test metrics.
7. Upload `prediction.csv`, request predictions, and download the CSV. Then try `prediction_labeled.csv` to obtain labeled performance metrics.
8. Ask for a comparison and a report of the completed work.
9. Download needed artifacts and select **End & delete**. Confirm the session becomes inaccessible and its temporary workspace is removed.

Also download both the task script and standalone experiment ZIP from a completed run. Extract the ZIP on a machine with Python 3.12, follow its `README.txt` to install dependencies, and run `python run.py`. Verify outputs appear in `data/out/`. The ZIP includes the input data for that task; evaluation and prediction packages also include the fitted model. Check that another session cannot download these artifacts. Rebuild the sandbox image after updating the export/runtime code so new runs produce the ZIP.

Check public health separately:

```bash
curl -fsS https://ml.your-domain.com/api/health
```

Also verify that the application remains available after disconnecting SSH. The public site, inference request, and sandbox execution must all work; a rendered frontend alone does not establish end-to-end success.

## 15. Record containment and prepare the submission

Use the application's **Run 3-second containment check** button. It dispatches an infinite loop to a disposable container and should report that the loop was terminated at the timeout and the container removed. Run a normal task afterward to show that the application remains usable.

Include that moment in the recorded demo, together with the actual ML workflow. Prepare the public application URL, public GitHub repository, demo video, and required submission materials. The README describes the use case and Vultr architecture; this file documents reproducible deployment. Follow the event page for the final submission form and requirements.

## Updating an existing deployment

Save or commit your own work before updating. Transfer the new code or pull a committed version from GitHub. Stop the backend before changing runtime files; restarting intentionally ends active sessions.

```bash
sudo systemctl stop ml-assistant
cd ~/ML_Assistant
git pull --ff-only
cd backend
uv sync --locked
cd ../frontend
npm ci
npm test
npm run build
cd ..
docker build -f sandbox/Dockerfile -t ml-assistant-sandbox:latest .
cd backend
uv run pytest -m 'not docker' -q
uv run pytest -m docker -q
```

If every check passes:

```bash
sudo systemctl start ml-assistant
sudo systemctl status ml-assistant --no-pager
curl -fsS https://ml.your-domain.com/api/health
```

Rebuild the sandbox whenever `sandbox/` or `backend/app/task_templates/` changes. Rebuild the frontend whenever its source or public configuration changes. Restart the backend after backend configuration changes. If updating the service template, reapply the path/user adaptation and run `sudo systemctl daemon-reload` before restarting.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| SSH connection times out | VM state, public IP, TCP 22 firewall rules, and whether your workplace network blocks SSH. Ping alone does not verify SSH. |
| SSH reports `Permission denied (publickey)` | The actual VM username, selected public key, matching local private key, and private-key permissions. |
| Inference returns 401 | Copy the actual inference API key, not its label or the VM management key. Check for whitespace and confirm the subscription is active. |
| Docker returns a socket permission error | Reconnect after adding the operator to the Docker group; check `docker ps` without sudo. |
| `docker buildx build requires 1 argument` | Run the documented build command from the repository root and include the final dot. |
| Health says the sandbox is unavailable | Verify Docker is running and the configured `SANDBOX_IMAGE` exists. |
| Root URL returns a build instruction instead of the app | Build `frontend/dist/` with `npm run build`, then restart the backend. |
| Public site returns 502 | Check the backend service and `curl http://127.0.0.1:8000/api/health`; align Caddy and service ports. |
| HTTPS certificate fails | Check DNS A/AAAA records and public reachability of ports 80/443. |
| Session creation fails over plain HTTP | Use the public HTTPS URL. For temporary HTTP-only tunnel testing, `COOKIE_SECURE=false` is needed; restore true before public HTTPS deployment. |
| Training times out or runs out of memory | Use a smaller dataset, fewer trees, or fewer expanded features. Keep resource caps in place. |
| A change to a task template has no effect | Rebuild the sandbox image; restarting FastAPI alone does not update the image. |

Service diagnostics:

```bash
sudo journalctl -u ml-assistant -n 80 --no-pager
sudo journalctl -u caddy -n 80 --no-pager
docker ps
```

Avoid posting environment files, keys, or user data when sharing diagnostics. Session files are temporary: end-session and expiry cleanup delete them, failed cleanup is retried, and startup removes abandoned workspaces. This deployment does not provide session recovery, forensic disk erasure, or control over Vultr's handling of inference requests.
