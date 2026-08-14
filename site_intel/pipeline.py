"""The three chained calls.

    home → choose which pages to read → read them → write the dossier → extract fields

Each step feeds the next, and each one is a separate function so it can be
tested on its own against a scripted backend.

The guardrail worth pointing at: the model is asked to *choose* links, not to
produce them, and whatever it returns is checked against the list it was given.
A URL that was not on the list is dropped and counted. Models are good at
picking and quietly bad at transcribing — they drop a hyphen, pluralise a slug,
or complete a path that looked unfinished — and a 404 three steps later is a
miserable thing to debug.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from site_intel import prompts, tokens
from site_intel.config import ModelSpec
from site_intel.llm import ChatBackend, Usage, describe_error, parse_json_object
from site_intel.scraper import FetchError, Page, fetch, normalise

MAX_LINKS = 5
HOME_EXCERPT_CHARS = 4_000
PAGE_EXCERPT_CHARS = 3_000
#: The corpus is what the dossier call pays for; keep it predictable.
CORPUS_TOKEN_BUDGET = 12_000


@dataclass(frozen=True)
class Facts:
    """The structured extraction. Every field can legitimately be missing."""

    company: str | None = None
    sector: str | None = None
    size_hint: str | None = None
    offering: str | None = None
    customers: str | None = None
    tech: tuple[str, ...] = ()
    hiring: bool = False
    hiring_detail: str | None = None
    sales_angle: str | None = None
    confidence: str | None = None

    @classmethod
    def from_json(cls, payload: dict) -> Facts:
        """Tolerant by design: a missing field is missing, not a crash."""

        def text(key: str) -> str | None:
            value = payload.get(key)
            if value is None or isinstance(value, bool):
                return None
            value = str(value).strip()
            return value or None if value.lower() not in ("null", "none", "n/a") else None

        raw_tech = payload.get("tech")
        tech = (
            tuple(str(item).strip() for item in raw_tech if str(item).strip())
            if isinstance(raw_tech, list)
            else ()
        )
        return cls(
            company=text("company"),
            sector=text("sector"),
            size_hint=text("size_hint"),
            offering=text("offering"),
            customers=text("customers"),
            tech=tech,
            hiring=bool(payload.get("hiring")),
            hiring_detail=text("hiring_detail"),
            sales_angle=text("sales_angle"),
            confidence=text("confidence"),
        )


@dataclass(frozen=True)
class Selection:
    chosen: tuple[tuple[str, str], ...] = ()  # (kind, url)
    #: URLs the model returned that were never on the list it was given.
    invented: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)
    estimated_tokens: int = 0


@dataclass(frozen=True)
class Dossier:
    url: str
    company: str
    markdown: str
    facts: Facts
    pages_read: tuple[str, ...]
    usage: Usage
    estimated_tokens: int
    invented_links: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# progress events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    label: str
    detail: str = ""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Finished:
    dossier: Dossier


@dataclass(frozen=True)
class Failed:
    reason: str


Event = Step | TextDelta | Finished | Failed


# --------------------------------------------------------------------------
# step 1 -- choose
# --------------------------------------------------------------------------


def select_messages(page: Page, candidates: Sequence) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompts.SELECT_LINKS_SYSTEM},
        {"role": "user", "content": prompts.select_links_user(page, candidates)},
    ]


def dossier_messages(company: str, corpus: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompts.DOSSIER_SYSTEM},
        {"role": "user", "content": prompts.dossier_user(company, corpus)},
    ]


def facts_messages(dossier: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompts.FACTS_SYSTEM},
        {"role": "user", "content": prompts.facts_user(dossier)},
    ]


def select_links(
    page: Page, model: ModelSpec, backend: ChatBackend, limit: int = MAX_LINKS
) -> Selection:
    candidates = [link for link in page.links if link.same_domain]
    if not candidates:
        return Selection()

    messages = select_messages(page, candidates)
    estimated = tokens.count_messages(messages, model.key)
    reply, usage = backend.complete(messages, model, json_mode=True)

    payload = parse_json_object(reply) or {}
    offered = {link.url for link in candidates}
    chosen: list[tuple[str, str]] = []
    invented: list[str] = []

    for entry in payload.get("links") or []:
        if not isinstance(entry, dict):
            continue
        raw = str(entry.get("url") or "").strip()
        if not raw:
            continue
        try:
            url = normalise(raw)
        except FetchError:
            invented.append(raw)
            continue
        if url not in offered:
            # It was not on the list. Models transcribe URLs badly.
            invented.append(raw)
            continue
        kind = str(entry.get("type") or "página").strip()
        if url not in {existing for _, existing in chosen}:
            chosen.append((kind, url))

    return Selection(tuple(chosen[:limit]), tuple(invented), usage, estimated)


# --------------------------------------------------------------------------
# step 2 -- read
# --------------------------------------------------------------------------


def build_corpus(
    home: Page, selected: Sequence[tuple[str, str]], use_cache: bool = True
) -> tuple[str, list[str], list[str]]:
    """Return ``(corpus, pages_read, notes)``."""
    parts = [f"## Home ({home.url})\n\n{home.excerpt(HOME_EXCERPT_CHARS)}"]
    read = [home.url]
    notes: list[str] = []

    for kind, url in selected:
        try:
            page = fetch(url, use_cache=use_cache)
        except FetchError as error:
            notes.append(f"No pude leer {url}: {error}")
            continue
        parts.append(f"## {kind} ({url})\n\n{page.excerpt(PAGE_EXCERPT_CHARS)}")
        read.append(url)

    corpus = tokens.trim_to_tokens("\n\n".join(parts), CORPUS_TOKEN_BUDGET)
    return corpus, read, notes


# --------------------------------------------------------------------------
# step 3 and 4 -- write, then extract
# --------------------------------------------------------------------------


def write_dossier(
    company: str, corpus: str, model: ModelSpec, backend: ChatBackend
) -> Iterator[str]:
    yield from backend.stream(dossier_messages(company, corpus), model)


def extract_facts(
    dossier: str, model: ModelSpec, backend: ChatBackend
) -> tuple[Facts, Usage, int]:
    """Returns ``(facts, usage, estimated_prompt_tokens)``."""
    messages = facts_messages(dossier)
    estimated = tokens.count_messages(messages, model.key)
    reply, usage = backend.complete(messages, model, json_mode=True)
    payload = parse_json_object(reply)
    if payload is None:
        return Facts(), usage, estimated
    return Facts.from_json(payload), usage, estimated


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------


def run(
    url: str, model: ModelSpec, backend: ChatBackend, *, use_cache: bool = True
) -> Iterator[Event]:
    """Drive the pipeline end to end, reporting progress as it goes."""
    try:
        yield Step("Leyendo la home", url)
        home = fetch(url, use_cache=use_cache)
    except FetchError as error:
        yield Failed(str(error))
        return

    company = home.title.split("—")[0].split("|")[0].strip() or home.url
    notes: list[str] = []
    total = Usage()

    try:
        yield Step("Eligiendo qué páginas leer", f"{len(home.links)} links encontrados")
        selection = select_links(home, model, backend)
    except Exception as error:  # noqa: BLE001 - a provider failure is a message, not a stack trace
        yield Failed(describe_error(error))
        return

    total = total + selection.usage
    estimated = selection.estimated_tokens
    if selection.invented:
        notes.append(
            f"{len(selection.invented)} URL(s) que el modelo devolvió no estaban en la lista "
            f"y se descartaron: {', '.join(selection.invented[:3])}"
        )

    kinds = list(dict.fromkeys(kind for kind, _ in selection.chosen))
    yield Step("Leyendo las páginas elegidas", ", ".join(kinds) or "ninguna")
    corpus, pages_read, fetch_notes = build_corpus(home, selection.chosen, use_cache=use_cache)
    notes.extend(fetch_notes)

    # Estimated per call and accumulated, so it can be compared against the
    # provider's own prompt_tokens for the whole run.
    dossier_estimate = tokens.count_messages(dossier_messages(company, corpus), model.key)
    estimated += dossier_estimate

    yield Step("Escribiendo el dossier", f"~{dossier_estimate:,} tokens de entrada")
    markdown = ""
    try:
        for fragment in write_dossier(company, corpus, model, backend):
            markdown += fragment
            yield TextDelta(fragment)
    except Exception as error:  # noqa: BLE001
        yield Failed(describe_error(error))
        return
    total = total + backend.last_usage

    yield Step("Extrayendo campos estructurados")
    try:
        facts, facts_usage, facts_estimate = extract_facts(markdown, model, backend)
    except Exception as error:  # noqa: BLE001
        yield Failed(describe_error(error))
        return
    total = total + facts_usage
    estimated += facts_estimate

    yield Finished(
        Dossier(
            url=home.url,
            company=facts.company or company,
            markdown=markdown.strip(),
            facts=facts,
            pages_read=tuple(pages_read),
            usage=total,
            estimated_tokens=estimated,
            invented_links=selection.invented,
            notes=tuple(notes),
        )
    )
