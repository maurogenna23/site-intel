"""Settings and the model registry.

Small on purpose: this is a CLI, not a platform. What it does need is an
explicit statement of which models are usable right now, because half the point
of the tool is running the same pipeline through two of them and comparing.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
OUT_DIR = ROOT / "out"

# Anchored to the package: a relative default would follow the shell around and
# quietly write the cache wherever the command happened to be run.
load_dotenv(ROOT / ".env", override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#: A polite, honest identifier. Sites are entitled to know who is reading them.
USER_AGENT = "site-intel/0.1 (+https://github.com/maurogenna/site-intel)"
REQUEST_TIMEOUT = 15
#: Hard cap on downloaded bytes per page, before any parsing.
MAX_PAGE_BYTES = 2_000_000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    litellm_id: str
    label: str
    is_local: bool
    requires_env: str | None = None
    #: Whether the provider honours ``response_format={"type": "json_object"}``.
    supports_json_mode: bool = True

    @property
    def available(self) -> bool:
        if self.is_local:
            return ollama_is_running()
        return bool(os.getenv(self.requires_env or ""))


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt-4.1-mini", "openai/gpt-4.1-mini", "GPT-4.1 mini · OpenAI", False, "OPENAI_API_KEY"),
    ModelSpec("gpt-4.1-nano", "openai/gpt-4.1-nano", "GPT-4.1 nano · OpenAI", False, "OPENAI_API_KEY"),
    ModelSpec(
        "gemini-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "Gemini 3.1 Flash Lite · Google",
        False,
        "GOOGLE_API_KEY",
    ),
    ModelSpec("groq-oss", "groq/openai/gpt-oss-120b", "GPT-OSS 120B · Groq", False, "GROQ_API_KEY"),
    ModelSpec("llama3.2", "ollama_chat/llama3.2", "Llama 3.2 3B · local", True),
)

MODELS_BY_KEY = {model.key: model for model in MODELS}
DEFAULT_MODEL = "gpt-4.1-mini"


@lru_cache(maxsize=1)
def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_BASE_URL, timeout=0.7) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS_BY_KEY[key]
    except KeyError:
        raise SystemExit(
            f"Modelo desconocido: {key!r}. Disponibles: {', '.join(MODELS_BY_KEY)}"
        ) from None


def available_models() -> list[ModelSpec]:
    return [model for model in MODELS if model.available]


def describe_environment() -> str:
    lines = ["site-intel", ""]
    lines.append(f"Ollama en {OLLAMA_BASE_URL}: {'accesible' if ollama_is_running() else 'no accesible'}")
    lines.append("")
    lines.append("Modelos:")
    for model in MODELS:
        mark = "ok " if model.available else "-- "
        why = "" if model.available else f"  (falta {model.requires_env or 'Ollama corriendo'})"
        lines.append(f"  {mark}{model.label:<32}{why}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_environment())
