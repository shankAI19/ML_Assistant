import { config } from "../config.js";
import { request } from "../api.js";

export function wireUploads(onUpdate, onError) {
  document.querySelector("#upload-hint").textContent =
    `Up to ${config.maxUploadMb} MB · binary classification`;
  for (const kind of ["training", "prediction"]) {
    const input = document.querySelector(`#${kind}-file`);
    input.addEventListener("change", async () => {
      const file = input.files[0];
      if (!file) return;
      try {
        if (!file.name.toLowerCase().endsWith(".csv"))
          throw new Error("Choose a CSV file.");
        if (file.size > config.maxUploadMb * 1024 * 1024)
          throw new Error("This file exceeds the upload limit.");
        input.disabled = true;
        const form = new FormData();
        form.append("file", file);
        await request(`/uploads?kind=${kind}`, { method: "POST", body: form });
        document.querySelector(`#${kind}-status`).textContent =
          `${file.name} · uploaded`;
        await onUpdate();
      } catch (error) {
        onError(error);
      } finally {
        input.disabled = false;
        input.value = "";
      }
    });
  }
}
