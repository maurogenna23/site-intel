"""Fetching and cleaning. Every test runs against a local fixture: no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from site_intel import scraper
from site_intel.scraper import FetchError, Link, domain_of, normalise, parse

FIXTURE = Path(__file__).parent / "fixtures" / "company.html"
BASE = "https://kepler.dev"


@pytest.fixture
def page():
    return parse(FIXTURE.read_text(encoding="utf-8"), BASE)


# --------------------------------------------------------------------------
# urls
# --------------------------------------------------------------------------


def test_normalise_fills_in_what_people_leave_out() -> None:
    assert normalise("kepler.dev") == "https://kepler.dev/"
    assert normalise("HTTPS://Kepler.DEV/nosotros/") == "https://kepler.dev/nosotros"
    assert normalise("https://kepler.dev/a#seccion") == "https://kepler.dev/a"
    assert normalise("https://kepler.dev/buscar?q=1") == "https://kepler.dev/buscar?q=1"


def test_normalise_rejects_what_it_cannot_read() -> None:
    for bad, expected in [("", "vacía"), ("ftp://kepler.dev", "http"), ("https://", "No entiendo")]:
        with pytest.raises(FetchError, match=expected):
            normalise(bad)


def test_domain_ignores_www() -> None:
    assert domain_of("https://www.kepler.dev/x") == domain_of("https://kepler.dev/y") == "kepler.dev"


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


def test_metadata_is_extracted(page) -> None:
    assert page.title == "Talleres Kepler — Software a medida"
    assert page.description.startswith("Construimos software a medida")


def test_noise_never_reaches_the_model(page) -> None:
    for junk in ("window.analytics", ".hero", "color: red"):
        assert junk not in page.text


def test_repeated_chrome_is_dropped_from_the_text(page) -> None:
    """Nav and footer repeat on every page and drown the actual content."""
    assert "Talleres Kepler SRL" not in page.text
    assert "Software a medida para operaciones complejas" in page.text
    assert "catorce personas" in page.text


def test_chrome_can_be_kept_when_asked() -> None:
    kept = parse(FIXTURE.read_text(encoding="utf-8"), BASE, keep_chrome=True)
    assert "Talleres Kepler SRL" in kept.text


def test_excerpt_trims_at_a_word_boundary(page) -> None:
    excerpt = page.excerpt(60)
    assert len(excerpt) <= 61 and excerpt.endswith("…")
    assert not excerpt.rstrip("…").endswith(" ")
    assert page.excerpt(100_000) == page.excerpt(100_000)  # no truncation, no ellipsis
    assert "…" not in page.excerpt(100_000)


# --------------------------------------------------------------------------
# links -- the part the pipeline depends on
# --------------------------------------------------------------------------


def test_relative_links_are_resolved_in_code(page) -> None:
    """Not by the model: it cannot verify a URL it invents."""
    urls = {link.url for link in page.links}
    assert "https://kepler.dev/nosotros" in urls
    assert "https://kepler.dev/trabaja-con-nosotros" in urls  # no leading slash in the source


def test_anchor_text_is_kept_and_the_richest_one_wins(page) -> None:
    """/nosotros is linked twice, as "Nosotros" and as "Quiénes somos"."""
    nosotros = next(link for link in page.links if link.url.endswith("/nosotros"))
    assert nosotros.text == "Quiénes somos"


def test_nested_anchor_text_keeps_its_word_boundaries(page) -> None:
    """Without a separator this reads "ServiciosLo que hacemos"."""
    servicios = next(link for link in page.links if link.url.endswith("/servicios"))
    assert servicios.text == "Servicios Lo que hacemos"


def test_useless_links_are_dropped(page) -> None:
    urls = {link.url for link in page.links}
    assert not any(url.endswith("/privacidad") for url in urls)  # boilerplate
    assert not any("mailto" in url or "javascript" in url for url in urls)
    assert BASE + "/" not in urls  # the page linking to itself


def test_fragment_only_links_are_dropped(page) -> None:
    assert not any(link.url.endswith("#contacto") for link in page.links)


def test_off_domain_links_are_marked_not_removed(page) -> None:
    blog = next(link for link in page.links if "blog.kepler.dev" in link.url)
    assert blog.same_domain is False
    assert all(link.same_domain for link in page.links if link.url.startswith(f"{BASE}/"))


def test_link_renders_as_url_plus_context() -> None:
    assert str(Link("https://a.com/x", "Sobre nosotros", True)) == "https://a.com/x — Sobre nosotros"
    assert str(Link("https://a.com/x", "", True)) == "https://a.com/x"


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scraper, "CACHE_DIR", tmp_path)
    assert scraper._read_cache(BASE) is None

    scraper._write_cache(BASE, "<html>hola</html>")
    assert scraper._read_cache(BASE) == "<html>hola</html>"
    assert scraper._read_cache("https://otra.com") is None


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scraper, "CACHE_DIR", tmp_path)
    scraper._write_cache(BASE, "<html></html>")
    scraper._cache_path(BASE).write_text("{ no es json", encoding="utf-8")
    assert scraper._read_cache(BASE) is None


def test_cached_pages_skip_the_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scraper, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        scraper.requests, "get", lambda *a, **k: pytest.fail("no debería salir a la red")
    )
    scraper._write_cache(BASE, FIXTURE.read_text(encoding="utf-8"))

    html, from_cache = scraper.download(BASE)
    assert from_cache and "Talleres Kepler" in html
