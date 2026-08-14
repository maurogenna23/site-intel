"""The command line.

    site-intel arniechat.com
    site-intel arniechat.com --model gemini-flash-lite --out out/arnie.md
    site-intel arniechat.com --compare gpt-4.1-mini,llama3.2
    site-intel --models

Exit codes: 0 fine, 1 the pipeline failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from site_intel import compare, pipeline, report
from site_intel.config import (
    DEFAULT_MODEL,
    MODELS_BY_KEY,
    OUT_DIR,
    ModelSpec,
    describe_environment,
    get_model,
)
from site_intel.llm import ChatBackend, default_backend
from site_intel.pipeline import Dossier, Failed, Finished, Step, TextDelta
from site_intel.scraper import domain_of, normalise

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _supports_colour(stream) -> bool:  # noqa: ANN001
    return hasattr(stream, "isatty") and stream.isatty()


class Printer:
    """Writes progress to stderr and the dossier to stdout, so piping works."""

    def __init__(self, quiet: bool = False, stream=None) -> None:  # noqa: ANN001
        self.quiet = quiet
        # Resolved on every write, never captured here: a default argument is
        # bound once at import, which pins the printer to whatever sys.stderr
        # was back then and ignores anyone who replaces it afterwards.
        self._stream = stream

    @property
    def stream(self):  # noqa: ANN201
        return self._stream or sys.stderr

    def _write(self, text: str, decoration: str) -> None:
        stream = self.stream
        colour = _supports_colour(stream)
        print(f"{decoration}{text}{RESET}" if colour else text, file=stream, flush=True)

    def step(self, text: str) -> None:
        if not self.quiet:
            self._write(f"▸ {text}", DIM)

    def note(self, text: str) -> None:
        self._write(text, BOLD)

    def error(self, text: str) -> None:
        print(f"error: {text}", file=sys.stderr, flush=True)


def default_output(url: str) -> Path:
    return OUT_DIR / f"{domain_of(normalise(url)).replace('.', '-')}.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="site-intel",
        description="Convierte la URL de una empresa en un dossier comercial.",
    )
    parser.add_argument("url", nargs="?", help="URL de la empresa (con o sin https://)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=sorted(MODELS_BY_KEY), help="modelo a usar"
    )
    parser.add_argument("--out", type=Path, help="archivo de salida (por defecto out/<dominio>.md)")
    parser.add_argument("--compare", help="corre el mismo pipeline en varios modelos: a,b,c")
    parser.add_argument("--no-cache", action="store_true", help="ignora el caché de HTML en disco")
    parser.add_argument("--quiet", action="store_true", help="sin líneas de progreso")
    parser.add_argument("--models", action="store_true", help="lista los modelos disponibles y sale")
    return parser


def run_one(
    url: str,
    model: ModelSpec,
    backend: ChatBackend,
    printer: Printer,
    *,
    use_cache: bool = True,
    echo: bool = True,
) -> tuple[Dossier | None, str, float]:
    """Drive the pipeline. Returns ``(dossier, error, seconds)``."""
    started = time.perf_counter()
    for event in pipeline.run(url, model, backend, use_cache=use_cache):
        if isinstance(event, Step):
            printer.step(event.label + (f" — {event.detail}" if event.detail else ""))
        elif isinstance(event, TextDelta) and echo:
            print(event.text, end="", flush=True)
        elif isinstance(event, Failed):
            return None, event.reason, time.perf_counter() - started
        elif isinstance(event, Finished):
            if echo:
                print()
            return event.dossier, "", time.perf_counter() - started
    return None, "El pipeline terminó sin producir nada.", time.perf_counter() - started


def run_comparison(
    args: argparse.Namespace, backend: ChatBackend, printer: Printer, check_credentials: bool = True
) -> int:
    """Same URL, same pipeline, several models, one table at the end."""
    keys = [key.strip() for key in args.compare.split(",") if key.strip()]
    models = [get_model(key) for key in keys]
    unavailable = [model for model in models if check_credentials and not model.available]
    if unavailable:
        printer.error(
            "sin credenciales para: " + ", ".join(model.label for model in unavailable)
        )
        return 1

    attempts: list[compare.Attempt] = []
    for model in models:
        printer.note(f"\n=== {model.label} ===")
        # The dossier is streamed to stderr here: stdout is for the table, and
        # printing four dossiers to stdout would bury it.
        dossier, error, seconds = run_one(
            args.url, model, backend, printer, use_cache=not args.no_cache, echo=False
        )
        attempts.append(compare.Attempt(model, dossier, error, seconds))
        if dossier is None:
            printer.error(error)
            continue
        destination = (args.out or default_output(args.url)).with_name(
            f"{(args.out or default_output(args.url)).stem}-{model.key}.md"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.render(dossier, model), encoding="utf-8")
        printer.note(report.summary_line(dossier, model, seconds))
        printer.note(f"escrito en {destination}")

    table = compare.render(normalise(args.url), attempts)
    comparison_path = (args.out or default_output(args.url)).with_name(
        f"{(args.out or default_output(args.url)).stem}-comparacion.md"
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(table, encoding="utf-8")

    print(f"\n{compare.table(attempts)}")
    printer.note(f"\ncomparación escrita en {comparison_path}")
    return 0 if any(attempt.ok for attempt in attempts) else 1


def main(argv: Sequence[str] | None = None, backend: ChatBackend | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.models:
        print(describe_environment())
        return 0
    if not args.url:
        build_parser().print_usage(sys.stderr)
        print("error: falta la URL (o usá --models)", file=sys.stderr)
        return 2

    # A caller that supplies its own backend has taken responsibility for the
    # transport, so the credential check does not apply to it. That is what the
    # tests do, and it keeps them runnable on a machine with no keys at all.
    check_credentials = backend is None
    backend = backend or default_backend()
    printer = Printer(quiet=args.quiet)

    if args.compare:
        return run_comparison(args, backend, printer, check_credentials)

    model = get_model(args.model)
    if check_credentials and not model.available:
        printer.error(
            f"{model.label} no está disponible: falta {model.requires_env or 'Ollama corriendo'}."
        )
        return 1

    dossier, error, seconds = run_one(
        args.url, model, backend, printer, use_cache=not args.no_cache
    )
    if dossier is None:
        printer.error(error)
        return 1

    destination = args.out or default_output(args.url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.render(dossier, model), encoding="utf-8")

    printer.note(f"\n{report.summary_line(dossier, model, seconds)}")
    printer.note(f"escrito en {destination}")
    for note in dossier.notes:
        printer.step(f"aviso: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
