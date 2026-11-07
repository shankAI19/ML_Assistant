import "./styles.css";
import { config } from "./config.js";
import { request } from "./api.js";
import { state } from "./state.js";
import { renderChat } from "./components/chat.js";
import { renderApproval } from "./components/approval.js";
import { renderResults, clearResults } from "./components/results.js";
import { wireUploads, renderUploads } from "./components/upload.js";
import { showResults } from "./layout.js";

document.title = config.appTitle;
document.querySelector("#app-title").textContent = config.appTitle;

function notify(error) {
  const notice = document.querySelector("#notice");
  notice.textContent = error.message || String(error);
  notice.hidden = false;
  if (error.status === 401) stop();
}

function stop() {
  state.stopped = true;
  clearTimeout(state.poll);
  state.session = null;
  state.signature = "";
  state.resultsSignature = "";
  document.querySelector("#messages").replaceChildren();
  document.querySelector("#message").value = "";
  renderUploads();
  clearResults();
  showResults(false);
  document.querySelector("#send-message").disabled = true;
  document.querySelector("#containment-demo").disabled = true;
  document.querySelector("#agent-status").textContent = "Session ended";
  document.querySelector("#approval-panel").hidden = true;
  document.querySelector("#end-session").textContent = "Start a new session";
}

async function update() {
  if (state.stopped) return;
  const expectedExperiment = state.session?.experiment_id;
  const snapshot = await request("/sessions/current");
  if (state.stopped || expectedExperiment !== state.session?.experiment_id)
    return;
  if (
    state.session?.experiment_id &&
    state.session.experiment_id !== snapshot.experiment_id
  ) {
    clearResults();
    showResults(false);
    document.querySelector("#notice").hidden = true;
    state.signature = "";
    state.resultsSignature = "";
  }
  state.session = snapshot;
  const signature = JSON.stringify([
    snapshot.messages,
    snapshot.status,
    snapshot.error,
    snapshot.pending,
  ]);
  if (signature !== state.signature) {
    state.signature = signature;
    renderChat(snapshot.messages, snapshot.busy);
    renderApproval(snapshot.pending, async (decision) => {
      try {
        await request("/approvals", {
          method: "POST",
          body: JSON.stringify(decision),
        });
        await update();
      } catch (error) {
        notify(error);
      }
    });
  }
  document.querySelector("#agent-status").textContent = snapshot.busy
    ? "Working…"
    : snapshot.pending
      ? "Your approval"
      : "Ready";
  document
    .querySelector("#agent-status")
    .classList.toggle("working", snapshot.busy);
  document.querySelector("#send-message").disabled =
    snapshot.busy || !!snapshot.pending;
  renderUploads();
  document.querySelector("#containment-demo").disabled =
    snapshot.busy || !!snapshot.pending;
  if (snapshot.error) notify(new Error(snapshot.error));
  const resultSignature = snapshot.runs.map((run) => run.id).join();
  if (resultSignature !== state.resultsSignature) {
    state.resultsSignature = resultSignature;
    showResults(snapshot.runs.length > 0);
    await renderResults(snapshot.runs);
  }
}

async function poll() {
  try {
    await update();
  } catch (error) {
    notify(error);
  }
  if (!state.stopped)
    state.poll = setTimeout(poll, state.session?.busy ? 1000 : 4000);
}

document
  .querySelector("#chat-form")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.querySelector("#message");
    const message = input.value.trim();
    if (!message || state.session?.busy || state.session?.pending) return;
    try {
      document.querySelector("#send-message").disabled = true;
      document.querySelector("#notice").hidden = true;
      await request("/chat", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
      input.value = "";
      await update();
    } catch (error) {
      notify(error);
      document.querySelector("#send-message").disabled = false;
    }
  });
document.querySelector("#message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!document.querySelector("#send-message").disabled)
      document.querySelector("#chat-form").requestSubmit();
  }
});
document.querySelectorAll("[data-prompt]").forEach((button) =>
  button.addEventListener("click", () => {
    document.querySelector("#message").value = button.dataset.prompt;
    document.querySelector("#message").focus();
  }),
);
document.querySelector("#end-session").addEventListener("click", async () => {
  if (state.stopped) {
    window.location.reload();
    return;
  }
  if (
    !window.confirm(
      "Delete this session and all uploaded data, models, and reports? Download what you need first.",
    )
  )
    return;
  try {
    const deletion = await request("/sessions/current", { method: "DELETE" });
    state.session = null;
    document.querySelector("#messages").replaceChildren();
    document.querySelector("#results").replaceChildren();
    document.querySelector("#training-status").textContent =
      "Session data deleted";
    document.querySelector("#message").value = "";
    stop();
    if (deletion.cleanup_pending) {
      notify(
        new Error(
          "Session ended. Cleanup is pending and will be retried automatically.",
        ),
      );
    }
  } catch (error) {
    notify(error);
  }
});
document
  .querySelector("#containment-demo")
  .addEventListener("click", async () => {
    if (
      !window.confirm(
        "Run an infinite loop in a disposable sandbox with a 3-second timeout?",
      )
    )
      return;
    try {
      await request("/containment-demo", { method: "POST" });
      await update();
    } catch (error) {
      notify(error);
    }
  });
wireUploads(update, notify);
try {
  state.session = await request("/sessions", { method: "POST" });
  renderUploads();
  const health = await request("/health");
  document.querySelector("#environment-status").textContent =
    `${health.sandbox_ready ? "Sandbox ready" : "Sandbox not ready"} · ${health.llm_configured ? "Inference configured" : "Inference key needed"}`;
  renderChat([], false);
  await poll();
} catch (error) {
  notify(error);
}
