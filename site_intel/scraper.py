"""Fetching and cleaning web pages.

Three decisions here are deliberate departures from the naive version:

* **Links keep their anchor text.** A bare list of URLs asks the model to guess
  from slugs; ``/quienes-somos`` with the text "Nuestro equipo" is a far
  stronger signal, for the same tokens.
* **Relative URLs are resolved in code**, with ``urljoin``. Asking a model to do
  string surgery it cannot verify is a way to get plausible, broken URLs.
* **Raw HTML is cached on disk.** Re-running the pipeline, or changing how the
  text is cleaned, costs no requests. It also makes the site a bit less annoyed.

``robots.txt`` is honoured. A tool that reads other people's sites at scale
should ask first.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from site_intel.config import CACHE_DIR, MAX_PAGE_BYTES, REQUEST_TIMEOUT, USER_AGENT

#: Chrome and everything else strip these before a human ever sees the page.
NOISE_TAGS = ("script", "style", "noscript", "svg", "img", "input", "form", "iframe")
#: Structural chrome that repeats on every page and drowns the actual content.
CHROME_TAGS = ("nav", "header", "footer", "aside")

SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "#")
#: Pages that never say anything about the company.
SKIP_PATTERNS = re.compile(
    r"/(privacy|privacidad|terms|terminos|cookies?|legal|login|signin|signup|cart|checkout)",
    re.IGNORECASE,
)


class FetchError(RuntimeError):
    """A page could not be read, phrased for the person running the command."""


@dataclass(frozen=True)
class Link:
    url: str
    text: str
    same_domain: bool

    def __str__(self) -> str:
        return f"{self.url} — {self.text}" if self.text else self.url


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    description: str
    text: str
    links: tuple[Link, ...] = ()
    from_cache: bool = False

    def excerpt(self, limit: int) -> str:
        """Title, description and body, trimmed at a word boundary."""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        parts.append(self.text)
        joined = "\n\n".join(part for part in parts if part)
        if len(joined) <= limit:
            return joined
        return joined[:limit].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------
# urls
# --------------------------------------------------------------------------


def normalise(url: str) -> str:
    """Add a scheme if missing, drop fragments, and strip a trailing slash."""
    candidate = url.strip()
    if not candidate:
        raise FetchError("La URL está vacía.")
    if not urlparse(candidate).scheme:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"Solo puedo leer http o https, no {parsed.scheme!r}.")
    if not parsed.netloc:
        raise FetchError(f"No entiendo la URL {url!r}.")

    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", parsed.query, ""))


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


# --------------------------------------------------------------------------
# robots.txt
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _robots(domain_root: str) -> urllib.robotparser.RobotFileParser | None:
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(urljoin(domain_root, "/robots.txt"))
    try:
        parser.read()
    except Exception:  # noqa: BLE001 - no robots.txt, or unreachable: assume allowed
        return None
    return parser


def may_fetch(url: str) -> bool:
    parsed = urlparse(url)
    parser = _robots(f"{parsed.scheme}://{parsed.netloc}")
    return True if parser is None else parser.can_fetch(USER_AGENT, url)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.json"


def _read_cache(url: str) -> str | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["html"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None  # a corrupt entry is a cache miss, not a crash


def _write_cache(url: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(
        json.dumps({"url": url, "fetched_at": datetime.now().isoformat(), "html": html}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# fetching and parsing
# --------------------------------------------------------------------------


def download(url: str, use_cache: bool = True) -> tuple[str, bool]:
    """Return ``(html, from_cache)``."""
    if use_cache:
        cached = _read_cache(url)
        if cached is not None:
            return cached, True

    if not may_fetch(url):
        raise FetchError(f"El robots.txt de {domain_of(url)} no permite leer {url}.")

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.Timeout:
        raise FetchError(f"{url} tardó más de {REQUEST_TIMEOUT} s en responder.") from None
    except requests.HTTPError as error:
        raise FetchError(f"{url} respondió {error.response.status_code}.") from None
    except requests.RequestException as error:
        raise FetchError(f"No pude llegar a {url}: {type(error).__name__}.") from None

    content_type = response.headers.get("Content-Type", "")
    if content_type and "html" not in content_type.lower():
        raise FetchError(f"{url} devolvió {content_type.split(';')[0]}, no HTML.")

    html = response.text[:MAX_PAGE_BYTES]
    if use_cache:
        _write_cache(url, html)
    return html, False


def parse(html: str, url: str, keep_chrome: bool = False) -> Page:
    """Turn HTML into a :class:`Page`. Pure: no network, easy to test."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    description = ""
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta and meta.get("content"):
        description = meta["content"].strip()

    links = _extract_links(soup, url)

    for tag in soup(list(NOISE_TAGS)):
        tag.decompose()
    if not keep_chrome:
        # Done after link extraction: the nav is where the interesting links are.
        for tag in soup(list(CHROME_TAGS)):
            tag.decompose()

    body = soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", body.get_text(separator="\n", strip=True))

    return Page(url=url, title=title, description=description, text=text, links=links)


def _extract_links(soup: BeautifulSoup, base_url: str) -> tuple[Link, ...]:
    base_domain = domain_of(base_url)
    seen: dict[str, Link] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(SKIP_SCHEMES):
            continue
        absolute = urljoin(base_url, href)
        try:
            absolute = normalise(absolute)
        except FetchError:
            continue
        if absolute == normalise(base_url) or SKIP_PATTERNS.search(urlparse(absolute).path):
            continue

        # The separator matters: nested spans otherwise concatenate into
        # "Reservas y turnosAgenda sola", which is worse signal than no text.
        text = " ".join(anchor.get_text(" ", strip=True).split())[:80]
        existing = seen.get(absolute)
        # Keep the most descriptive anchor text for a URL linked several times.
        if existing is None or len(text) > len(existing.text):
            seen[absolute] = Link(absolute, text, domain_of(absolute) == base_domain)

    return tuple(seen.values())


def fetch(url: str, use_cache: bool = True) -> Page:
    """Download and parse a page in one step."""
    clean = normalise(url)
    html, from_cache = download(clean, use_cache=use_cache)
    page = parse(html, clean)
    return Page(**{**page.__dict__, "from_cache": from_cache})
