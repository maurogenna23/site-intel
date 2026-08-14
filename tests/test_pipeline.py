"""The chained calls, against a scripted backend. No network, no API key, no cost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import ScriptedBackend, facts_reply, links_reply

from site_intel import pipeline, scraper
from site_intel.config import get_model
from site_intel.llm import Usage
from site_intel.pipeline import Facts, Failed, Finished, Step, TextDelta, select_links
from site_intel.scraper import parse

FIXTURE = Path(__file__).parent / "fixtures" / "company.html"
BASE = "https://kepler.dev"
MODEL = get_model("gpt-4.1-mini")


@pytest.fixture
def home():
    return parse(FIXTURE.read_text(encoding="utf-8"), BASE)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    """Every fetch returns the fixture; nothing leaves the machine."""
    pages: dict[str, int] = {}

    def fake_fetch(url: str, use_cache: bool = True):  # noqa: FBT001, FBT002
        pages[url] = pages.get(url, 0) + 1
        if url.endswith("/contacto"):  # a real link on the page that 404s
            raise scraper.FetchError(f"{url} respondió 404.")
        return parse(FIXTURE.read_text(encoding="utf-8"), url)

    monkeypatch.setattr(pipeline, "fetch", fake_fetch)
    return pages


# --------------------------------------------------------------------------
# step 1 -- the model chooses, it does not invent
# --------------------------------------------------------------------------


def test_only_links_that_were_offered_are_accepted(home) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros", f"{BASE}/no-existe")])
    selection = select_links(home, MODEL, backend)

    assert [url for _, url in selection.chosen] == [f"{BASE}/nosotros"]
    assert selection.invented == (f"{BASE}/no-existe",)


def test_a_mistranscribed_url_is_caught(home) -> None:
    """Models drop hyphens and pluralise slugs; a 404 three steps later is worse."""
    backend = ScriptedBackend([links_reply(f"{BASE}/trabajaconnosotros")])
    selection = select_links(home, MODEL, backend)
    assert selection.chosen == ()
    assert selection.invented == (f"{BASE}/trabajaconnosotros",)


def test_off_domain_links_are_never_even_offered(home) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros")])
    select_links(home, MODEL, backend)
    listing = backend.seen_messages[0][1]["content"]
    assert "blog.kepler.dev" not in listing
    assert "/nosotros" in listing


def test_anchor_text_reaches_the_model(home) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros")])
    select_links(home, MODEL, backend)
    assert "Quiénes somos" in backend.seen_messages[0][1]["content"]


def test_json_mode_is_requested_for_the_structured_steps(home) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros")])
    select_links(home, MODEL, backend)
    assert backend.seen_json_mode == [True]


def test_selection_is_capped(home) -> None:
    urls = [link.url for link in home.links if link.same_domain]
    backend = ScriptedBackend([links_reply(*urls)])
    assert len(select_links(home, MODEL, backend, limit=2).chosen) == 2


def test_duplicate_choices_are_collapsed(home) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros", f"{BASE}/nosotros")])
    assert len(select_links(home, MODEL, backend).chosen) == 1


def test_a_reply_that_is_not_json_is_not_fatal(home) -> None:
    backend = ScriptedBackend(["perdón, no puedo hacer eso"])
    selection = select_links(home, MODEL, backend)
    assert selection.chosen == () and selection.invented == ()


def test_json_wrapped_in_prose_is_still_read(home) -> None:
    """Local models answer with a code fence and a sentence in front."""
    fenced = f"Claro, acá va:\n```json\n{links_reply(f'{BASE}/nosotros')}\n```"
    assert select_links(home, MODEL, ScriptedBackend([fenced])).chosen[0][1] == f"{BASE}/nosotros"


def test_a_page_with_no_internal_links_skips_the_call() -> None:
    bare = parse("<html><body><p>hola</p></body></html>", BASE)
    backend = ScriptedBackend([])  # would raise if called
    assert select_links(bare, MODEL, backend).chosen == ()


# --------------------------------------------------------------------------
# the extraction
# --------------------------------------------------------------------------


def test_facts_are_parsed() -> None:
    facts = Facts.from_json(json.loads(facts_reply()))
    assert facts.company == "Talleres Kepler"
    assert facts.tech == ("Python", "PostgreSQL")
    assert facts.hiring is True


def test_missing_fields_stay_missing_instead_of_becoming_strings() -> None:
    facts = Facts.from_json({"company": "X", "sector": None, "size_hint": "null", "tech": None})
    assert facts.company == "X"
    assert facts.sector is None
    assert facts.size_hint is None, "the string 'null' is not a company size"
    assert facts.tech == ()
    assert facts.hiring is False


def test_extraction_survives_a_reply_that_is_not_json() -> None:
    facts, usage, estimated = pipeline.extract_facts("ficha", MODEL, ScriptedBackend(["no sé"]))
    assert facts == Facts()
    assert usage.total_tokens > 0  # the call still cost something
    assert estimated > 0


def test_the_estimate_covers_every_call_not_just_one(offline) -> None:
    """Comparing one call's estimate against three calls' bill is not a comparison."""
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros"), facts_reply()])
    dossier = next(e for e in _events(backend) if isinstance(e, Finished)).dossier

    selection = select_links(
        parse(FIXTURE.read_text(encoding="utf-8"), BASE),
        MODEL,
        ScriptedBackend([links_reply(f"{BASE}/nosotros")]),
    )
    assert dossier.estimated_tokens > selection.estimated_tokens


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def _events(backend, **kwargs):  # noqa: ANN001
    return list(pipeline.run(BASE, MODEL, backend, **kwargs))


def test_the_whole_pipeline(offline) -> None:
    backend = ScriptedBackend(
        [links_reply(f"{BASE}/nosotros"), facts_reply()],
        stream_text="## Qué hacen\n\nSistemas internos.",
    )
    events = _events(backend)

    assert [type(event) for event in events].count(Step) == 5
    assert "".join(e.text for e in events if isinstance(e, TextDelta)).strip().startswith("## Qué hacen")

    dossier = next(e for e in events if isinstance(e, Finished)).dossier
    assert dossier.company == "Talleres Kepler"
    assert dossier.pages_read == (BASE, f"{BASE}/nosotros")
    assert dossier.facts.hiring is True
    assert dossier.estimated_tokens > 0


def test_usage_adds_up_across_the_three_calls(offline) -> None:
    each = Usage(prompt_tokens=1000, completion_tokens=100, cost_usd=0.001)
    backend = ScriptedBackend([links_reply(f"{BASE}/nosotros"), facts_reply()], usage=each)
    dossier = next(e for e in _events(backend) if isinstance(e, Finished)).dossier

    assert dossier.usage.prompt_tokens == 3000
    assert dossier.usage.completion_tokens == 300
    assert dossier.usage.cost_usd == pytest.approx(0.003)


def test_a_page_that_fails_is_noted_and_the_rest_continues(offline) -> None:
    backend = ScriptedBackend(
        [links_reply(f"{BASE}/contacto", f"{BASE}/nosotros"), facts_reply()],
    )
    dossier = next(e for e in _events(backend) if isinstance(e, Finished)).dossier

    assert f"{BASE}/contacto" not in dossier.pages_read
    assert f"{BASE}/nosotros" in dossier.pages_read
    assert any("404" in note for note in dossier.notes)


def test_invented_links_are_reported_in_the_dossier(offline) -> None:
    backend = ScriptedBackend([links_reply(f"{BASE}/inventado"), facts_reply()])
    dossier = next(e for e in _events(backend) if isinstance(e, Finished)).dossier
    assert dossier.invented_links == (f"{BASE}/inventado",)
    assert any("no estaban en la lista" in note for note in dossier.notes)


def test_an_unreachable_home_stops_before_spending_anything(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline, "fetch", lambda *a, **k: (_ for _ in ()).throw(scraper.FetchError("no responde"))
    )
    backend = ScriptedBackend([])  # would raise if the pipeline called it
    events = _events(backend)
    assert isinstance(events[-1], Failed) and "no responde" in events[-1].reason


def test_a_provider_outage_is_a_message_not_a_stack_trace(offline) -> None:
    class RateLimitError(Exception):
        pass

    backend = ScriptedBackend(error=RateLimitError("Limit 8000"))
    events = _events(backend)
    assert isinstance(events[-1], Failed)
    assert "límite de uso" in events[-1].reason
