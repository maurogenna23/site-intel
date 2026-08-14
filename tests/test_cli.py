"""The command line and the rendered report. No network."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fakes import ScriptedBackend, facts_reply, links_reply

from site_intel import cli, pipeline, report
from site_intel.config import get_model
from site_intel.llm import Usage
from site_intel.pipeline import Dossier, Facts
from site_intel.scraper import parse

FIXTURE = Path(__file__).parent / "fixtures" / "company.html"
BASE = "https://kepler.dev"
MODEL = get_model("gpt-4.1-mini")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        pipeline, "fetch", lambda url, use_cache=True: parse(FIXTURE.read_text("utf-8"), url)
    )


def a_dossier(**overrides) -> Dossier:  # noqa: ANN003
    base = {
        "url": BASE,
        "company": "Talleres Kepler",
        "markdown": "## Qué hacen\n\nSistemas internos.",
        "facts": Facts(company="Talleres Kepler", sector="software", tech=("Python",), hiring=True),
        "pages_read": (BASE, f"{BASE}/nosotros"),
        "usage": Usage(prompt_tokens=7000, completion_tokens=900, cached_tokens=3968, cost_usd=0.0033),
        "estimated_tokens": 6990,
    }
    return Dossier(**{**base, **overrides})


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def test_report_records_what_produced_it() -> None:
    text = report.render(a_dossier(), MODEL, generated_at=datetime(2026, 8, 14, 21, 30))
    assert "# Talleres Kepler" in text
    assert "2026-08-14 21:30" in text and MODEL.label in text
    assert f"<{BASE}/nosotros>" in text  # the pages actually read
    assert "3,968 cacheados" in text
    assert "0.3300 ¢" in text


def test_report_shows_the_estimate_against_the_bill() -> None:
    text = report.render(a_dossier(), MODEL)
    assert "estimado 6,990 vs real 7,000 (-0.1%)" in text


def test_missing_facts_render_as_a_dash_not_as_none() -> None:
    table = report.facts_table(Facts(company="X"))
    assert "| Empresa | X |" in table
    assert "| Rubro | — |" in table
    assert "None" not in table


def test_hiring_without_detail_still_says_so() -> None:
    assert "sí, sin detalle" in report.facts_table(Facts(hiring=True))
    assert "| Contrata | — |" in report.facts_table(Facts(hiring=False))


def test_warnings_are_not_swept_under_the_rug() -> None:
    text = report.render(a_dossier(notes=("No pude leer /contacto: 404.",)), MODEL)
    assert "## Advertencias" in text and "404" in text
    assert "## Advertencias" not in report.render(a_dossier(), MODEL)


def test_unpriced_runs_say_so() -> None:
    text = report.render(a_dossier(usage=Usage(prompt_tokens=10, cost_usd=None)), MODEL)
    assert "n/d" in text


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def test_defaults() -> None:
    args = cli.build_parser().parse_args(["kepler.dev"])
    assert args.url == "kepler.dev" and args.model == "gpt-4.1-mini"
    assert args.out is None and not args.no_cache


def test_unknown_model_is_rejected_before_any_work() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["kepler.dev", "--model", "gpt-9"])


def test_output_path_is_derived_from_the_domain() -> None:
    assert cli.default_output("https://www.kepler.dev/algo").name == "kepler-dev.md"


def test_no_url_is_a_usage_error() -> None:
    assert cli.main([], backend=ScriptedBackend([])) == 2


def test_models_flag_needs_no_url(capsys) -> None:
    assert cli.main(["--models"], backend=ScriptedBackend([])) == 0
    assert "Modelos:" in capsys.readouterr().out


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_writes_the_dossier_and_reports_success(offline, tmp_path: Path, capsys) -> None:
    destination = tmp_path / "kepler.md"
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros"), facts_reply()])

    code = cli.main([BASE, "--out", str(destination)], backend=backend)
    assert code == 0

    written = destination.read_text(encoding="utf-8")
    assert "# Talleres Kepler" in written and "## Datos estructurados" in written
    assert "## Qué hacen" in capsys.readouterr().out  # streamed to stdout as it arrives


def test_a_failure_exits_nonzero_and_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    from site_intel.scraper import FetchError

    monkeypatch.setattr(
        pipeline, "fetch", lambda *a, **k: (_ for _ in ()).throw(FetchError("no responde"))
    )
    destination = tmp_path / "nada.md"
    assert cli.main([BASE, "--out", str(destination)], backend=ScriptedBackend([])) == 1
    assert not destination.exists()


def test_progress_goes_to_stderr_so_stdout_can_be_piped(offline, tmp_path: Path, capsys) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros"), facts_reply()])
    cli.main([BASE, "--out", str(tmp_path / "x.md")], backend=backend)

    captured = capsys.readouterr()
    assert "Leyendo la home" in captured.err
    assert "Leyendo la home" not in captured.out


def test_quiet_silences_progress_but_not_the_result(offline, tmp_path: Path, capsys) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros"), facts_reply()])
    cli.main([BASE, "--out", str(tmp_path / "x.md"), "--quiet"], backend=backend)

    captured = capsys.readouterr()
    assert "▸" not in captured.err
    assert "escrito en" in captured.err
