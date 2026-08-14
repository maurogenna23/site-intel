"""The comparison. Completeness is the column that matters."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import ScriptedBackend, facts_reply, links_reply

from site_intel import cli, compare, pipeline
from site_intel.compare import Attempt, filled_fields, rows, table
from site_intel.config import get_model
from site_intel.llm import Usage
from site_intel.pipeline import Dossier, Facts
from site_intel.scraper import parse

FIXTURE = Path(__file__).parent / "fixtures" / "company.html"
BASE = "https://kepler.dev"
GPT = get_model("gpt-4.1-mini")
LLAMA = get_model("llama3.2")


def a_dossier(facts: Facts | None = None, **overrides) -> Dossier:  # noqa: ANN003
    base = {
        "url": BASE,
        "company": "Kepler",
        "markdown": "## Qué hacen\n\nUna dos tres cuatro cinco.",
        "facts": facts if facts is not None else Facts(company="Kepler", sector="software"),
        "pages_read": (BASE,),
        "usage": Usage(prompt_tokens=5000, completion_tokens=500, cost_usd=0.002),
        "estimated_tokens": 4990,
    }
    return Dossier(**{**base, **overrides})


# --------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------


def test_completeness_counts_what_actually_came_back() -> None:
    assert filled_fields(Facts()) == 0
    assert filled_fields(Facts(company="X")) == 1

    full = Facts(
        company="X",
        sector="y",
        size_hint="14",
        offering="z",
        customers="w",
        tech=("Python",),
        hiring_detail="dos backend",
        sales_angle="por ahí",
        confidence="alta",
    )
    assert filled_fields(full) == 9


def test_empty_strings_do_not_count_as_extracted() -> None:
    assert filled_fields(Facts(company="   ", tech=())) == 0


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_ranked_by_completeness_not_by_price() -> None:
    """A cheap model returning four nulls has not done the job."""
    thorough = Attempt(GPT, a_dossier(Facts(company="X", sector="y", size_hint="z")), seconds=9.0)
    cheap = Attempt(LLAMA, a_dossier(Facts(company="X")), seconds=1.0)

    assert [row[0] for row in rows([cheap, thorough])] == [GPT.label, LLAMA.label]


def test_failures_sink_and_are_labelled() -> None:
    ok = Attempt(GPT, a_dossier(), seconds=2.0)
    broken = Attempt(LLAMA, None, "se cayó", 0.5)
    listed = rows([broken, ok])
    assert listed[0][0] == GPT.label
    assert listed[1][-1] == "falló"


def test_invented_urls_are_a_column(  ) -> None:
    attempt = Attempt(LLAMA, a_dossier(invented_links=("https://x.com/no", "https://x.com/tampoco")))
    assert rows([attempt])[0][2] == "2"
    assert rows([Attempt(GPT, a_dossier())])[0][2] == "0"


def test_table_is_markdown_with_a_row_per_model() -> None:
    text = table([Attempt(GPT, a_dossier()), Attempt(LLAMA, a_dossier())])
    assert text.count("\n") == 3  # header, separator, two rows
    assert "| Modelo |" in text


def test_render_says_what_each_model_chose_to_read() -> None:
    text = compare.render(
        BASE,
        [
            Attempt(GPT, a_dossier(pages_read=(BASE, f"{BASE}/nosotros"))),
            Attempt(LLAMA, None, "se cayó"),
        ],
    )
    assert f"- <{BASE}/nosotros>" in text
    assert "Falló: se cayó" in text
    assert "Campos que no pudo extraer" in text  # the fixture facts are partial


# --------------------------------------------------------------------------
# through the cli
# --------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        pipeline, "fetch", lambda url, use_cache=True: parse(FIXTURE.read_text("utf-8"), url)
    )


def test_compare_writes_one_file_per_model_plus_the_table(offline, tmp_path: Path, capsys) -> None:
    backend = ScriptedBackend(
        [links_reply(f"{BASE}/nosotros"), facts_reply(), links_reply(f"{BASE}/nosotros"), facts_reply()]
    )
    destination = tmp_path / "kepler.md"

    code = cli.main(
        [BASE, "--out", str(destination), "--compare", "gpt-4.1-mini,gpt-4.1-nano"], backend=backend
    )
    assert code == 0

    assert (tmp_path / "kepler-gpt-4.1-mini.md").exists()
    assert (tmp_path / "kepler-gpt-4.1-nano.md").exists()
    assert "| Modelo |" in (tmp_path / "kepler-comparacion.md").read_text(encoding="utf-8")
    assert "| Modelo |" in capsys.readouterr().out  # the table goes to stdout


def test_compare_refuses_models_without_credentials(offline, tmp_path: Path, monkeypatch) -> None:
    """The real entry point builds its own backend, and then credentials matter."""
    monkeypatch.setattr(cli, "default_backend", lambda: ScriptedBackend([]))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    code = cli.main([BASE, "--out", str(tmp_path / "x.md"), "--compare", "gpt-4.1-mini,groq-oss"])
    assert code == 1
    assert not list(tmp_path.glob("*.md"))
