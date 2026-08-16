"""Keep the hosted quickstart clean and aligned with the authoritative V6 script."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "getting_started" / "pricing_and_iv.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "offline_quickstart_colab.ipynb"
README_PATH = REPO_ROOT / "README.md"
QUICKSTART_PATH = REPO_ROOT / "docs" / "getting_started.md"
COLAB_URL = (
    "https://colab.research.google.com/github/ArturSepp/VanillaOptionPricers/"
    "blob/main/notebooks/offline_quickstart_colab.ipynb"
)
STABLE_QUICKSTART_URL = (
    "https://vanillaoptionpricers.readthedocs.io/en/stable/getting_started.html"
)
CANONICAL_SOURCE_URL = (
    "https://github.com/ArturSepp/VanillaOptionPricers/blob/main/"
    "examples/getting_started/pricing_and_iv.py"
)


def test_colab_notebook_is_a_clean_mirror_of_authoritative_example() -> None:
    """The hosted notebook adds setup only; its workflow cannot drift from V6."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("Colab drift contract applies only to a repository checkout")
    if not NOTEBOOK_PATH.is_file():
        pytest.fail(f"approved Colab notebook is missing: {NOTEBOOK_PATH}")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    workflow_cells = [
        cell
        for cell in code_cells
        if "v6-source" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(workflow_cells) == 1
    assert "".join(workflow_cells[0]["source"]) == EXAMPLE_PATH.read_text(encoding="utf-8")

    setup_source = "\n".join("".join(cell["source"]) for cell in code_cells[:-1])
    assert "https://pypi.org/simple" in setup_source
    assert "vanilla-option-pricers==" not in setup_source
    assert "git+" not in setup_source
    assert "version('vanilla-option-pricers')" in setup_source
    assert "vop.__file__" in setup_source
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    assert all(not cell.get("attachments") for cell in cells)

    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    assert STABLE_QUICKSTART_URL in markdown
    assert CANONICAL_SOURCE_URL in markdown
    assert COLAB_URL in QUICKSTART_PATH.read_text(encoding="utf-8")
    assert COLAB_URL in README_PATH.read_text(encoding="utf-8")
