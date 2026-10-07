"""Sphinx configuration for the VanillaOptionPricers documentation."""

import os
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
project = "vanilla-option-pricers"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
release = metadata["version"]

extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The publishers reject automated link checks even though these canonical records are live.
linkcheck_ignore = [
    r"https://doi\.org/10\.1080/14697688\.2024\.2364804",
    r"https://ssrn\.com/abstract=4606748",
]
# GitHub line anchors drift when the referenced sibling repository changes; still check each file.
linkcheck_anchors_ignore_for_url = [
    r"https://github\.com/ArturSepp/StochVolModels/blob/main/",
]
# Avoid tripping anonymous-host rate limits by checking external URLs sequentially.
linkcheck_workers = 1

html_theme = "furo"


def _consolidate_stable(url: str) -> str:
    """Return the canonical base URL with the moving ``stable`` alias replaced by ``latest``.

    Read the Docs builds ``stable`` from the newest release tag and ``latest`` from ``main``, so
    both serve the same pages. Left alone, each copy names itself canonical and search engines see
    every page twice. Numbered versions keep their own canonical URL.
    """
    return re.sub(r"(\.readthedocs\.io/en/)stable(/|$)", r"\1latest\2", url)


html_baseurl = _consolidate_stable(
    os.environ.get(
        "READTHEDOCS_CANONICAL_URL",
        "https://vanillaoptionpricers.readthedocs.io/en/latest/",
    )
)
html_title = "vanilla-option-pricers - Numba-vectorised BSM and Bachelier pricing"
html_short_title = "vanilla-option-pricers"
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/VanillaOptionPricers/",
    "source_branch": "main",
    "source_directory": "docs/",
}


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the site root, rather than ``index.html``, as the homepage canonical URL."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)
