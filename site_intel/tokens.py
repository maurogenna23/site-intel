"""Counting tokens before paying for them.

``tiktoken`` gives an exact count for OpenAI models and a good approximation for
everyone else — the tokenizers differ, but not by enough to change a decision
about whether a prompt is too big. The point is to know the size of what you are
about to send *before* sending it, and afterwards to compare that estimate with
what the provider actually billed.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from site_intel.llm import Message

#: Per-message overhead in the chat format: role, separators, priming.
MESSAGE_OVERHEAD = 4
FALLBACK_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def _encoding(model_key: str):  # noqa: ANN202 - tiktoken's type is not worth importing eagerly
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_key)
    except KeyError:
        # Anything that is not an OpenAI id: same family of tokenizer, close
        # enough to size a prompt with.
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def count_text(text: str, model_key: str = "gpt-4.1-mini") -> int:
    return len(_encoding(model_key).encode(text))


def count_messages(messages: Sequence[Message], model_key: str = "gpt-4.1-mini") -> int:
    encoding = _encoding(model_key)
    return sum(
        len(encoding.encode(str(message.get("content") or ""))) + MESSAGE_OVERHEAD
        for message in messages
    )


def trim_to_tokens(text: str, limit: int, model_key: str = "gpt-4.1-mini") -> str:
    """Cut a text to fit a token budget, on a token boundary.

    Character-based truncation is a guess that gets worse the less English the
    text is; Spanish and accented words cost more tokens per character.
    """
    encoding = _encoding(model_key)
    tokens = encoding.encode(text)
    if len(tokens) <= limit:
        return text
    return encoding.decode(tokens[:limit]) + "…"


def accuracy(estimated: int, actual: int) -> str:
    """How far the pre-flight estimate was from the bill, for the report."""
    if not actual:
        return "sin datos del proveedor"
    delta = (estimated - actual) / actual
    return f"estimado {estimated:,} vs real {actual:,} ({delta:+.1%})"
