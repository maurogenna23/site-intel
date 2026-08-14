"""Running the same pipeline through several models and reporting the difference.

Sequential rather than parallel, on purpose: the HTML cache means the second
model pays nothing for fetching, and running them one at a time keeps free-tier
rate limits out of the measurement.

The interesting column is not cost, it is **completeness**: how many of the nine
structured fields actually came back filled. A model that is cheap and fast and
returns four nulls has not done the job, and that is invisible if you only
compare tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from site_intel.config import ModelSpec
from site_intel.pipeline import Dossier, Facts

#: The fields the extraction is asked for, in the order they are reported.
FACT_FIELDS = (
    "company",
    "sector",
    "size_hint",
    "offering",
    "customers",
    "tech",
    "hiring_detail",
    "sales_angle",
    "confidence",
)


@dataclass(frozen=True)
class Attempt:
    model: ModelSpec
    dossier: Dossier | None = None
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.dossier is not None


def filled_fields(facts: Facts) -> int:
    """How many of the nine fields came back with something in them."""
    count = 0
    for name in FACT_FIELDS:
        value = getattr(facts, name)
        if isinstance(value, tuple):
            count += bool(value)
        else:
            count += bool(value and str(value).strip())
    return count


HEADERS = ("Modelo", "Páginas", "Inventadas", "Campos", "Tokens", "Costo", "Seg", "Palabras")


def rows(attempts: Sequence[Attempt]) -> list[list[str]]:
    """Best completeness first; failures last."""
    ordered = sorted(
        attempts,
        key=lambda attempt: (
            not attempt.ok,
            -(filled_fields(attempt.dossier.facts) if attempt.ok else 0),
        ),
    )
    out = []
    for attempt in ordered:
        if not attempt.ok:
            out.append([attempt.model.label, "—", "—", "—", "—", "—", f"{attempt.seconds:.1f}", "falló"])
            continue
        dossier = attempt.dossier
        usage = dossier.usage
        cost = "n/d" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"
        out.append(
            [
                attempt.model.label,
                str(len(dossier.pages_read)),
                str(len(dossier.invented_links)) if dossier.invented_links else "0",
                f"{filled_fields(dossier.facts)}/{len(FACT_FIELDS)}",
                f"{usage.prompt_tokens:,}",
                cost,
                f"{attempt.seconds:.1f}",
                str(len(dossier.markdown.split())),
            ]
        )
    return out


def table(attempts: Sequence[Attempt]) -> str:
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    lines += ["| " + " | ".join(row) + " |" for row in rows(attempts)]
    return "\n".join(lines)


def render(url: str, attempts: Sequence[Attempt]) -> str:
    """The comparison file: the table, then what each model chose to read."""
    parts = [
        f"# Comparación · {url}",
        "",
        "El mismo pipeline en varios modelos. **Campos** es cuántos de los nueve",
        "campos estructurados volvieron con contenido: un modelo barato que devuelve",
        "cuatro nulls no hizo el trabajo, y eso no se ve mirando tokens.",
        "",
        table(attempts),
    ]

    for attempt in attempts:
        parts += ["", f"## {attempt.model.label}"]
        if not attempt.ok:
            parts += ["", f"Falló: {attempt.error}"]
            continue
        parts += ["", "Páginas que eligió leer:", ""]
        parts += [f"- <{page}>" for page in attempt.dossier.pages_read]
        if attempt.dossier.invented_links:
            parts += [
                "",
                "URLs que devolvió y no estaban en la lista "
                f"(descartadas): {', '.join(attempt.dossier.invented_links)}",
            ]
        missing = [
            name for name in FACT_FIELDS if not getattr(attempt.dossier.facts, name)
        ]
        if missing:
            parts += ["", f"Campos que no pudo extraer: {', '.join(missing)}"]
    return "\n".join(parts) + "\n"
