"""The backend's only environment reader. Never export these settings to a container."""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.vultrinference.com/v1"
    model: str = "qwen3.8-flash-next"
    temperature: float = Field(0.2, ge=0, le=2)
    top_p: float = Field(0.2, gt=0, le=1)
    max_tokens: int = Field(4096, ge=256, le=16384)
    llm_timeout: int = Field(60, ge=5, le=300)
    repair_retries: int = Field(1, ge=0, le=2)
    sandbox_image: str = "ml-assistant-sandbox:latest"
    docker_host: str = (
        "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
    )
    sandbox_timeout: int = Field(120, ge=1, le=600)
    sandbox_memory_mb: int = Field(2048, ge=256, le=8192)
    sandbox_cpus: float = Field(1, gt=0, le=4)
    sandbox_pids: int = Field(128, ge=32, le=512)
    sandbox_tmpfs_mb: int = Field(256, ge=64, le=1024)
    max_upload_mb: int = Field(25, ge=1, le=100)
    max_output_mb: int = Field(64, ge=1, le=128)
    session_temp_root: Path = REPO_ROOT / ".runtime" / "sessions"
    idle_timeout: int = Field(1800, ge=60)
    max_lifetime: int = Field(7200, ge=60)
    cleanup_interval: int = Field(60, ge=1, le=300)
    max_sessions: int = Field(10, ge=1, le=50)
    session_storage_mb: int = Field(256, ge=64, le=1024)
    secure_cookie: bool = False
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)


ENV_FIELDS = {
    "VULTR_INFERENCE_API_KEY": "api_key",
    "VULTR_INFERENCE_BASE_URL": "base_url",
    "VULTR_INFERENCE_MODEL": "model",
    "LLM_TEMPERATURE": "temperature",
    "LLM_TOP_P": "top_p",
    "LLM_MAX_OUTPUT_TOKENS": "max_tokens",
    "LLM_REQUEST_TIMEOUT_SECONDS": "llm_timeout",
    "AGENT_MAX_REPAIR_RETRIES": "repair_retries",
    "SANDBOX_IMAGE": "sandbox_image",
    "SANDBOX_DOCKER_HOST": "docker_host",
    "SANDBOX_TIMEOUT_SECONDS": "sandbox_timeout",
    "SANDBOX_MEMORY_MB": "sandbox_memory_mb",
    "SANDBOX_CPUS": "sandbox_cpus",
    "SANDBOX_PIDS_LIMIT": "sandbox_pids",
    "SANDBOX_TMPFS_MB": "sandbox_tmpfs_mb",
    "BACKEND_MAX_UPLOAD_MB": "max_upload_mb",
    "SANDBOX_MAX_OUTPUT_MB": "max_output_mb",
    "SESSION_TEMP_ROOT": "session_temp_root",
    "SESSION_IDLE_TIMEOUT_SECONDS": "idle_timeout",
    "SESSION_MAX_LIFETIME_SECONDS": "max_lifetime",
    "SESSION_CLEANUP_INTERVAL_SECONDS": "cleanup_interval",
    "MAX_ACTIVE_SESSIONS": "max_sessions",
    "SESSION_MAX_STORAGE_MB": "session_storage_mb",
    "COOKIE_SECURE": "secure_cookie",
    "BACKEND_HOST": "host",
    "BACKEND_PORT": "port",
}


def load_settings() -> Settings:
    # No external tracing of user data, regardless of inherited developer settings.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    values = {**dotenv_values(BACKEND_ROOT / "back.env"), **os.environ}
    data = {field: values[key] for key, field in ENV_FIELDS.items() if values.get(key) is not None}
    settings = Settings.model_validate(data)
    root = settings.session_temp_root
    settings.session_temp_root = (root if root.is_absolute() else REPO_ROOT / root).resolve()
    # Destructive cleanup is restricted to this dedicated app directory.
    if settings.session_temp_root != (REPO_ROOT / ".runtime" / "sessions").resolve():
        raise ValueError("SESSION_TEMP_ROOT must be .runtime/sessions under this repository")
    return settings
