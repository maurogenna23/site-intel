"""Repo hygiene: the things that quietly rot between commits."""

from __future__ import annotations

from site_intel.config import MODELS, ROOT


def test_env_example_documents_every_key_the_registry_needs() -> None:
    documented = (ROOT / ".env.example").read_text(encoding="utf-8")
    for model in MODELS:
        if model.requires_env:
            assert model.requires_env in documented, f"{model.key} necesita {model.requires_env}"


def test_secrets_and_cache_are_not_committed() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert ".cache/" in ignored


def test_readme_is_not_a_stub() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme) > 2000
    for section in ("## The pipeline", "## Running it", "## Tests"):
        assert section in readme


def test_the_examples_the_readme_links_to_exist() -> None:
    """A README that links to a missing file is worse than one without links."""
    for name in ("huggingface-co.md", "arniechat-com-comparacion.md"):
        example = ROOT / "out" / "ejemplo" / name
        assert example.is_file(), f"falta el ejemplo {name}"
        assert len(example.read_text(encoding="utf-8")) > 500
