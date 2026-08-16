"""Sphinx configuration for the VanillaOptionPricers documentation."""

import os
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

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://vanillaoptionpricers.readthedocs.io/en/latest/",
)
html_title = "vanilla-option-pricers - Numba-vectorised BSM and Bachelier pricing"
html_short_title = "vanilla-option-pricers"
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/VanillaOptionPricers/",
    "source_branch": "main",
    "source_directory": "docs/",
}
