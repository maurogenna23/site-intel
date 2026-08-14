# site-intel

[![ci](https://github.com/maurogenna23/site-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/maurogenna23/site-intel/actions/workflows/ci.yml)

Give it a company URL, get back a sales-ready dossier in Markdown: what they do,
who they sell to, how they work, what signals are worth acting on, and an angle
for the first message — plus the same information as structured fields for a CRM.

```bash
site-intel huggingface.co
```

Built to explore what happens when you chain LLM calls over messy, real-world
input: the model decides what to read, reads it, writes the analysis, and then
extracts fields from its own analysis. Three calls, each one feeding the next,
each one able to be wrong in a different way.

[An example dossier](out/ejemplo/huggingface-co.md) ·
[an example model comparison](out/ejemplo/arniechat-com-comparacion.md)

---

## The pipeline

```
home  →  choose which pages to read  →  read them  →  write the dossier  →  extract fields
         (JSON mode, one-shot)                        (streamed)            (JSON mode)
```

Every step is a separate function driven by a `ChatBackend` protocol, so the
whole chain runs against a scripted fake in the tests — no network, no key.

## The guardrail, and why it earns its place

The model is asked to **choose** links, never to produce them, and whatever it
returns is checked against the list it was given. Anything that was not on the
list is dropped and reported.

This is not theoretical. Running the same site through four models:

| Model | Pages read | Invented URLs | Fields | Tokens | Cost | Sec |
|---|--:|--:|--:|--:|--:|--:|
| GPT-4.1 mini · OpenAI | 6 | 0 | 7/9 | 7,177 | 0.3268 ¢ | 17.8 |
| Gemini 3.1 Flash Lite · Google | 6 | 0 | 7/9 | 7,660 | 0.3454 ¢ | 34.8 |
| Llama 3.2 3B · local | **1** | **3** | 7/9 | 2,901 | 0.0000 ¢ | 16.9 |
| GPT-4.1 nano · OpenAI | 5 | 1 | 6/9 | 6,292 | 0.0981 ¢ | 9.2 |

Llama 3.2 returned three URLs that were never offered to it:
`/nosotros`, `/trabaja-con-nosotros`, and — literally — the string
`"No encontrado"`. The first two are pages a company site usually has; that
site has neither. It answered from what a website *ought* to look like rather
than from the list in front of it, and without the check the pipeline would
have made three requests to 404s and written a dossier from the home page while
believing it had read four pages.

GPT-4.1 nano made a milder version of the same mistake, returning the home page
URL, which is excluded from the candidates on purpose.

**The trap in that table**: llama filled 7 of 9 fields from a single page. Field
count measures whether the extraction *produced* values, not whether they are
*true* — a small model with one sixth of the material will still fill the form.
Completeness is a floor, not a verdict.

## What else running it taught me

**tiktoken is accurate enough to trust.** The pre-flight estimate landed within
**0.1%** of the provider's own count (7,213 estimated vs 7,222 billed), across
different sites. Knowing the size of a prompt before sending it is cheap.

That number was wrong at first, and interestingly so: the estimate covered one
call and was being compared against the bill for three, which read as a −20.6%
error in tiktoken. The tool was fine; the accounting was not.

**The HTML cache pays twice.** It saves the requests, and because the corpus
comes out byte-identical, the provider's prompt cache kicks in on a re-run: the
same dossier went from 0.44 ¢ to 0.33 ¢ with 3,968 cached tokens.

**Anchor text is free signal.** Links are handed to the model as
`/automatizar-reservas — Reservas y turnos: agenda sola, sincronizada con tu
calendario` rather than as bare slugs. Same token budget, much better choices.
Getting there needed a separator fix: nested `<span>`s were concatenating into
`"Reservas y turnosAgenda sola"`.

## Design notes

- **Relative URLs are resolved with `urljoin`**, not by the model. String
  surgery it cannot verify produces plausible, broken links.
- **Nav and footer are stripped from the text but only after link extraction** —
  the nav is where the interesting links live, and the footer is the same on
  every page.
- **`robots.txt` is honoured.** A tool that reads other people's sites should ask.
- **JSON parsing is tolerant**: providers without a real JSON mode wrap the
  object in a code fence and put a sentence in front of it. Being strict there
  means a local model can never be compared against a cloud one, which is half
  the point of the tool.
- **Missing is missing.** A field the model did not find stays `None`; the string
  `"null"` is not a company size, and a dossier with holes is more useful than
  one with inventions.
- **The report records what produced it**: model, pages actually read, tokens,
  cost, and a warnings section when a page failed. A dossier that hides two 404s
  is worse than one that admits them.

## Running it

```bash
git clone <this repo> && cd site-intel
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env      # add at least OPENAI_API_KEY
site-intel --models       # what is usable right now
site-intel huggingface.co
```

```bash
site-intel arniechat.com --model gemini-flash-lite --out out/arnie.md
site-intel arniechat.com --compare gpt-4.1-mini,gpt-4.1-nano,llama3.2
site-intel arniechat.com --no-cache --quiet
```

Progress goes to stderr and the dossier to stdout, so `site-intel x.com --quiet >
dossier.md` works. Optional and free: `GOOGLE_API_KEY`
([AI Studio](https://aistudio.google.com/api-keys)) and `GROQ_API_KEY`
([Groq](https://console.groq.com/keys)). Local models need
[Ollama](https://ollama.com) with `ollama pull llama3.2`.

A dossier costs between 0.1 ¢ and 0.6 ¢ depending on the model and how much the
site has to say.

## Tests

```bash
pytest
```

Sixty-odd tests, **no API key, no network, no cost**. The scraper runs against a
local HTML fixture; the pipeline runs against a scripted backend, which is what
makes it possible to test invented URLs, mistranscribed slugs, replies that are
not JSON, JSON wrapped in prose, pages that 404 mid-run, and provider outages
without spending anything.

---

Written while working through Ed Donner's
[LLM Engineering](https://github.com/ed-donner/llm_engineering) course — the
ideas come from week 1, the code is my own.
