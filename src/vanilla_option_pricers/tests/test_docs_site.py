"""Search-facing contracts of the documentation site: titles, canonical URLs and descriptions."""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = REPO_ROOT / "docs"
LATEST = "https://vanillaoptionpricers.readthedocs.io/en/latest/"

pytestmark = pytest.mark.skipif(
    not (DOCS / "conf.py").is_file(),
    reason="documentation sources are absent from installed wheels",
)


def _load_conf() -> dict:
    """Run ``docs/conf.py`` and return its namespace."""
    pytest.importorskip("tomllib", reason="docs/conf.py reads pyproject.toml with tomllib")
    return runpy.run_path(str(DOCS / "conf.py"))


@pytest.mark.parametrize(
    ("service_url", "canonical_url"),
    [
        # stable and latest serve the same pages, so both name latest as canonical
        ("https://vanillaoptionpricers.readthedocs.io/en/stable/", LATEST),
        (LATEST, LATEST),
        (
            "https://vanillaoptionpricers.readthedocs.io/en/2.2.0/",
            "https://vanillaoptionpricers.readthedocs.io/en/2.2.0/",
        ),
    ],
)
def test_stable_builds_name_latest_as_canonical(monkeypatch, service_url, canonical_url) -> None:
    """A ``stable`` build points its canonical URLs at ``latest``; numbered versions keep theirs."""
    monkeypatch.setenv("READTHEDOCS_CANONICAL_URL", service_url)
    assert _load_conf()["html_baseurl"] == canonical_url


def test_every_page_states_a_meta_description() -> None:
    """Every page states, in its MyST front matter, the description that search results show."""
    pages = sorted(DOCS.glob("*.md"))
    assert pages
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert text.startswith("---\nmyst:\n  html_meta:\n    description: "), page.name


def test_built_pages_carry_short_titles_and_a_root_homepage_canonical(
    monkeypatch, tmp_path
) -> None:
    """Furo would end every title with html_title and canonicalise the homepage as index.html."""
    for module in ("sphinx", "furo", "myst_parser"):
        pytest.importorskip(module)
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)
    conf = _load_conf()
    templates = [str(DOCS / path) for path in conf.get("templates_path", [])]
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "import runpy\n"
        f"_site = runpy.run_path({str(DOCS / 'conf.py')!r})\n"
        "extensions = ['myst_parser']\n"
        "html_theme = 'furo'\n"
        f"templates_path = {templates!r}\n"
        "for _key in ('project', 'html_title', 'html_baseurl'):\n"
        "    globals()[_key] = _site[_key]\n"
        "setup = _site.get('setup')\n",
        encoding="utf-8",
    )
    (source / "index.md").write_text("# Home\n\n```{toctree}\npricing\n```\n", encoding="utf-8")
    (source / "pricing.md").write_text("# Pricing and Greeks\n\nText.\n", encoding="utf-8")
    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-b", "html", str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    def head(name: str) -> str:
        """Return the head of a built page."""
        return (output / f"{name}.html").read_text(encoding="utf-8").split("</head>")[0]

    assert re.findall(r"<title>(.*?)</title>", head("index")) == [conf["html_title"]]
    assert re.findall(r"<title>(.*?)</title>", head("pricing")) == [
        "Pricing and Greeks - vanilla-option-pricers"
    ]
    assert f'<link rel="canonical" href="{LATEST}"' in head("index")
    assert f'<link rel="canonical" href="{LATEST}pricing.html"' in head("pricing")
