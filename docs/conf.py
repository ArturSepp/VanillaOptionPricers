"""Sphinx configuration for the VanillaOptionPricers documentation."""

import os
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin
from xml.etree import ElementTree

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


# Stable and latest may describe different source revisions; each build keeps its URL.
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://vanillaoptionpricers.readthedocs.io/en/latest/",
)
html_title = f"vanilla-option-pricers {release} - Numba-vectorised BSM and Bachelier pricing"
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



class _CanonicalHead(HTMLParser):
    """Read canonical and indexing metadata from the HTML actually being published."""

    def __init__(self):
        super().__init__()
        self.canonicals = []
        self.noindex = False
        self.redirect = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href"))
        if tag == "meta" and attributes.get("name", "").lower() == "robots":
            self.noindex |= "noindex" in attributes.get("content", "").lower()
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            self.redirect = True


def _write_canonical_sitemap(app, exception) -> None:
    """Offer only rendered, self-canonical pages from this build's own channel."""
    if exception is not None or app.builder.name != "html" or not app.config.html_baseurl:
        return
    base = app.config.html_baseurl.rstrip("/") + "/"
    urls = set()
    for docname in app.env.found_docs:
        head = _CanonicalHead()
        file = Path(app.builder.get_outfilename(docname))
        head.feed(file.read_text(encoding="utf-8").split("</head>")[0])
        if head.noindex or head.redirect or len(head.canonicals) != 1:
            continue
        uri = app.builder.get_target_uri(docname)
        expected = urljoin(base, quote("" if uri == "index.html" else uri, safe="/%"))
        if head.canonicals == [expected]:
            urls.add(expected)
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    root = ElementTree.Element(f"{{{namespace}}}urlset")
    for url in sorted(urls):
        entry = ElementTree.SubElement(root, f"{{{namespace}}}url")
        ElementTree.SubElement(entry, f"{{{namespace}}}loc").text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / "sitemap.xml", encoding="utf-8", xml_declaration=True,
        default_namespace=namespace,
    )


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)
    app.connect("build-finished", _write_canonical_sitemap)
