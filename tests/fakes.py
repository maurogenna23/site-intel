"""A scripted backend, so the pipeline can be tested without a network."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

from site_intel.llm import Message, Usage

DEFAULT_USAGE = Usage(prompt_tokens=1000, completion_tokens=120, cost_usd=0.0005)


class ScriptedBackend:
    """Replays canned replies and records what it was asked."""

    def __init__(
        self,
        completions: Sequence[str] | None = None,
        stream_text: str = "## Qué hacen\n\nSoftware a medida.",
        usage: Usage | None = None,
        error: Exception | None = None,
    ) -> None:
        self._completions = list(completions or [])
        self._stream_text = stream_text
        self._usage = usage or DEFAULT_USAGE
        self._error = error
        self.seen_messages: list[list[Message]] = []
        self.seen_json_mode: list[bool] = []
        self._last_usage = Usage()

    def complete(
        self, messages: Sequence[Message], model, *, json_mode: bool = False  # noqa: ANN001
    ) -> tuple[str, Usage]:
        if self._error:
            raise self._error
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_json_mode.append(json_mode)
        if not self._completions:
            raise AssertionError("the pipeline asked for more completions than the script provides")
        return self._completions.pop(0), self._usage

    def stream(self, messages: Sequence[Message], model) -> Iterator[str]:  # noqa: ANN001
        if self._error:
            raise self._error
        self.seen_messages.append([dict(m) for m in messages])
        for word in self._stream_text.split(" "):
            yield word + " "
        self._last_usage = self._usage

    @property
    def last_usage(self) -> Usage:
        return self._last_usage


def links_reply(*urls: str, kind: str = "sobre la empresa") -> str:
    return json.dumps({"links": [{"type": kind, "url": url} for url in urls]})


def facts_reply(**overrides) -> str:  # noqa: ANN003
    payload = {
        "company": "Talleres Kepler",
        "sector": "software a medida",
        "size_hint": "14 personas",
        "offering": "sistemas internos",
        "customers": "logística y retail",
        "tech": ["Python", "PostgreSQL"],
        "hiring": True,
        "hiring_detail": "dos backend",
        "sales_angle": "entrar por las integraciones",
        "confidence": "alta",
    }
    payload.update(overrides)
    return json.dumps(payload)
