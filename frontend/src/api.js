import { config } from "./config.js";

export async function request(path, options = {}) {
  const headers = { ...options.headers };
  if (typeof options.body === "string")
    headers["Content-Type"] = "application/json";
  const response = await fetch(`${config.apiBaseUrl}${path}`, {
    ...options,
    headers,
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((e) => e.msg).join("; ")
      : detail;
    const error = new Error(message || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export const artifactUrl = (runId, filename) =>
  `${config.apiBaseUrl}/artifacts/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`;
export const codeUrl = (runId) =>
  `${config.apiBaseUrl}/code/${encodeURIComponent(runId)}`;
