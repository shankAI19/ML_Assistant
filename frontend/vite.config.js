import { defineConfig } from "vite";
import { parse } from "dotenv";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const path = fileURLToPath(new URL("./front.env", import.meta.url));
const file = existsSync(path) ? parse(readFileSync(path)) : {};
const setting = (key, fallback) => process.env[key] ?? file[key] ?? fallback;
// An explicit allowlist prevents accidental frontend exposure of backend credentials.
const publicConfig = {
  appTitle: setting("FRONTEND_APP_TITLE", "ML Assistant"),
  apiBaseUrl: setting("FRONTEND_API_BASE_URL", "/api"),
  maxUploadMb: Number(setting("FRONTEND_MAX_UPLOAD_MB", "25")),
};
if (
  !publicConfig.apiBaseUrl.startsWith("/") ||
  publicConfig.apiBaseUrl.startsWith("//")
) {
  throw new Error(
    "FRONTEND_API_BASE_URL must be a same-origin path such as /api",
  );
}
if (
  !Number.isFinite(publicConfig.maxUploadMb) ||
  publicConfig.maxUploadMb < 1
) {
  throw new Error("FRONTEND_MAX_UPLOAD_MB must be a positive number");
}
export default defineConfig({
  envDir: false,
  envPrefix: [],
  define: { __APP_CONFIG__: JSON.stringify(publicConfig) },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: setting("FRONTEND_DEV_API_TARGET", "http://127.0.0.1:8000"),
        changeOrigin: false,
      },
    },
    fs: { deny: ["**/*.env", "**/*.env.*", "**/.git/**"] },
  },
  build: { sourcemap: false, chunkSizeWarningLimit: 6000 },
});
