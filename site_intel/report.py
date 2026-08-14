"""Rendering a dossier to Markdown.

The file is meant to be committed, diffed and read six months later, so it
records what produced it: which model, which pages were actually read, what it
cost, and anything that went wrong on the way. A dossier that hides the fact
that two of five pages 404'd is worse than one that says so.
"""

from __future__ import annotations

from datetime import datetime

from site_intel import tokens
from site_intel.config import ModelSpec
from site_intel.pipeline import Dossier, Facts

FACT_LABELS = (
    ("company", "Empresa"),
    ("sector", "Rubro"),
    ("size_hint", "Tamaño"),
    ("offering", "Qué vende"),
    ("customers", "A quién"),
    ("tech", "Stack"),
    ("hiring_detail", "Contrata"),
    ("sales_angle", "Ángulo"),
    ("confidence", "Confianza"),
)


def facts_table(facts: Facts) -> str:
    rows = ["| Campo | Valor |", "|---|---|"]
    for key, label in FACT_LABELS:
        value = getattr(facts, key)
        if key == "tech":
            value = ", ".join(value) if value else None
        if key == "hiring_detail" and not value:
            value = "sí, sin detalle" if facts.hiring else None
        rows.append(f"| {label} | {value if value else '—'} |")
    return "\n".join(rows)


def render(dossier: Dossier, model: ModelSpec, generated_at: datetime | None = None) -> str:
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    usage = dossier.usage
    cost = "n/d" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"

    parts = [
        f"# {dossier.company}",
        "",
        f"<{dossier.url}> · {stamp} · {model.label}",
        "",
        dossier.markdown,
        "",
        "## Datos estructurados",
        "",
        facts_table(dossier.facts),
        "",
        "## Páginas leídas",
        "",
        *[f"- <{url}>" for url in dossier.pages_read],
    ]

    if dossier.notes:
        parts += ["", "## Advertencias", "", *[f"- {note}" for note in dossier.notes]]

    parts += [
        "",
        "---",
        "",
        f"{usage.prompt_tokens:,} tokens de entrada · {usage.completion_tokens:,} de salida"
        + (f" · {usage.cached_tokens:,} cacheados" if usage.cached_tokens else "")
        + f" · {cost}",
        "",
        f"Estimación previa con tiktoken: {tokens.accuracy(dossier.estimated_tokens, usage.prompt_tokens)}",
    ]
    return "\n".join(parts) + "\n"


def summary_line(dossier: Dossier, model: ModelSpec, seconds: float) -> str:
    """The one line printed to the terminal when it is done."""
    usage = dossier.usage
    cost = "n/d" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"
    cached = f", {usage.cached_tokens:,} cacheados" if usage.cached_tokens else ""
    return (
        f"{model.label} · {len(dossier.pages_read)} páginas · "
        f"{usage.prompt_tokens:,} in{cached} / {usage.completion_tokens:,} out · "
        f"{cost} · {seconds:.1f} s"
    )
