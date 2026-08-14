"""The model gateway.

Everything that talks to a provider goes through :class:`ChatBackend`, so the
pipeline can be driven by a scripted fake in the tests without a network.

Two things this layer owns that the pipeline should not have to think about:
accounting (tokens, cache hits, real cost from LiteLLM's price map) and parsing
JSON out of a reply that was supposed to be JSON and sometimes is not quite.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from site_intel.config import ModelSpec

Message = dict[str, str]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    #: ``None`` when the provider reports no price we can trust.
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        cost = (
            None
            if self.cost_usd is None and other.cost_usd is None
            else (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        )
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            cost,
        )


class ChatBackend(Protocol):
    def complete(
        self, messages: Sequence[Message], model: ModelSpec, *, json_mode: bool = False
    ) -> tuple[str, Usage]:
        """One request, one answer."""

    def stream(
        self, messages: Sequence[Message], model: ModelSpec
    ) -> Iterator[str]:
        """Yield text fragments. ``last_usage`` holds the accounting afterwards."""

    @property
    def last_usage(self) -> Usage:
        """Accounting for the most recent :meth:`stream`."""


class LiteLLMBackend:
    """Real calls. ``litellm`` is imported lazily because it is slow to import."""

    def __init__(self) -> None:
        self._configured = False
        self._last_usage = Usage()

    def _litellm(self):  # noqa: ANN202 - third-party module
        import litellm

        if not self._configured:
            # Providers reject parameters they do not know; dropping them beats
            # branching per provider.
            litellm.drop_params = True
            litellm.suppress_debug_info = True
            self._configured = True
        return litellm

    def _accounting(self, response, model: ModelSpec) -> Usage:  # noqa: ANN001
        litellm = self._litellm()
        raw = getattr(response, "usage", None)
        details = getattr(raw, "prompt_tokens_details", None)
        if model.is_local:
            cost: float | None = 0.0
        else:
            try:
                cost = float(litellm.completion_cost(completion_response=response))
            except Exception:  # noqa: BLE001 - unknown model id in the price map
                cost = None
        return Usage(
            prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
            cached_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            cost_usd=cost,
        )

    def complete(
        self, messages: Sequence[Message], model: ModelSpec, *, json_mode: bool = False
    ) -> tuple[str, Usage]:
        litellm = self._litellm()
        kwargs: dict[str, object] = {"model": model.litellm_id, "messages": list(messages)}
        if json_mode and model.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or "", self._accounting(response, model)

    def stream(self, messages: Sequence[Message], model: ModelSpec) -> Iterator[str]:
        litellm = self._litellm()
        chunks = []
        for chunk in litellm.completion(
            model=model.litellm_id,
            messages=list(messages),
            stream=True,
            # Without this a streamed response carries no usage at all.
            stream_options={"include_usage": True},
        ):
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content

        try:
            rebuilt = litellm.stream_chunk_builder(chunks, messages=list(messages))
            self._last_usage = self._accounting(rebuilt, model)
        except Exception:  # noqa: BLE001 - accounting must never break the output
            self._last_usage = Usage(cost_usd=0.0 if model.is_local else None)

    @property
    def last_usage(self) -> Usage:
        return self._last_usage


def default_backend() -> ChatBackend:
    return LiteLLMBackend()


# --------------------------------------------------------------------------
# tolerant JSON
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(text: str) -> dict | None:
    """Pull an object out of a reply that was asked for JSON.

    Providers without a real JSON mode wrap it in a code fence, or add a
    sentence before it. Being strict here means a local model can never be
    compared against a cloud one, which is half the point of the tool.
    """
    if not text or not text.strip():
        return None

    candidate = _FENCE.sub("", text.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def describe_error(error: Exception) -> str:
    """Turn a provider exception into something worth showing a person."""
    name = type(error).__name__
    detail = str(error).replace("\n", " ")[:180]
    if "RateLimit" in name:
        return f"El proveedor cortó por límite de uso. Esperá unos segundos. — {detail}"
    if "NotFound" in name:
        return f"El proveedor no reconoce ese modelo; puede estar deprecado. — {detail}"
    if "Auth" in name or "PermissionDenied" in name:
        return f"Credenciales rechazadas. Revisá la API key en .env. — {detail}"
    if "Timeout" in name or "APIConnection" in name or "ServiceUnavailable" in name:
        return f"No pude llegar al proveedor. — {detail}"
    return f"Falló la llamada al modelo ({name}). — {detail}"
