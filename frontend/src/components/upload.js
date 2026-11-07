import { config } from "../config.js";
import { request } from "../api.js";
import { state } from "../state.js";

export function renderUploads() {
  const session = state.session;
  document.querySelector("#csv-file").disabled =
    state.stopped || state.uploading || !session || !!session.busy;
  document.querySelector("#training-status").textContent = state.stopped
    ? "Session ended"
    : session?.dataset_uploaded
      ? `Training: ${session.training_filename || "CSV uploaded"}`
      : "No training CSV uploaded";
  const prediction = document.querySelector("#prediction-status");
  prediction.hidden = !session?.prediction_uploaded;
  prediction.textContent = session?.prediction_uploaded
    ? `Prediction/test: ${session.prediction_filename || "CSV uploaded"}${session.finalized ? "" : " (queued until finalization)"}`
    : "";
  document.querySelector("#upload-hint").textContent = session?.busy
    ? "Task running. Uploads resume when it finishes."
    : `Training or prediction/test data · up to ${config.maxUploadMb} MB per CSV`;
}

function choosePurpose(file) {
  const dialog = document.querySelector("#upload-dialog");
  document.querySelector("#upload-filename").textContent = file.name;
  document.querySelector("#training-replacement-note").hidden =
    !state.session?.dataset_uploaded;
  dialog.returnValue = "";
  const decision = new Promise((resolve) =>
    dialog.addEventListener("close", () => resolve(dialog.returnValue), {
      once: true,
    }),
  );
  dialog.showModal();
  const preferred = state.session?.dataset_uploaded ? "prediction" : "training";
  dialog.querySelector(`button[value="${preferred}"]`).focus();
  return decision;
}

export function wireUploads(onUpdate, onError) {
  const input = document.querySelector("#csv-file");
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file || state.uploading || state.stopped) return;
    state.uploading = true;
    renderUploads();
    try {
      if (!file.name.toLowerCase().endsWith(".csv"))
        throw new Error("Choose a CSV file.");
      if (file.size > config.maxUploadMb * 1024 * 1024)
        throw new Error("This file exceeds the upload limit.");
      const kind = await choosePurpose(file);
      if (kind !== "training" && kind !== "prediction") return;
      if (state.stopped) return;
      const replace = kind === "training" && !!state.session?.dataset_uploaded;
      const form = new FormData();
      form.append("file", file);
      await request(
        `/uploads?kind=${kind}${replace ? "&replace_training=true" : ""}`,
        { method: "POST", body: form },
      );
      await onUpdate();
    } catch (error) {
      onError(error);
    } finally {
      state.uploading = false;
      input.value = "";
      renderUploads();
    }
  });
  renderUploads();
}
