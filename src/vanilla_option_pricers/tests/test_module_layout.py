"""Contracts for repository-only runnable examples."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
LEGACY_DISPATCHERS = {"LocalTest", "LocalTests", "local_test", "run_local_test"}


def _tree(path: Path) -> ast.Module:
    """Parse one Python module."""
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _is_main_guard(node: ast.AST) -> bool:
    """Return whether a node is an executable main guard."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _main_calls_selected_local(node: ast.If) -> bool:
    """Return whether the main guard contains only ``run_local(local=Locals.*)``."""
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
        return False
    call = node.body[0].value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "run_local"
        and not call.args
        and len(call.keywords) == 1
        and call.keywords[0].arg == "local"
        and isinstance(call.keywords[0].value, ast.Attribute)
        and isinstance(call.keywords[0].value.value, ast.Name)
        and call.keywords[0].value.value.id == "Locals"
    )


@pytest.mark.skipif(not EXAMPLES_ROOT.exists(), reason="examples are absent from installed wheels")
def test_repository_examples_use_current_dispatcher_api() -> None:
    """Every runnable example uses ``Locals`` and ``run_local(local=...)``."""
    failures: list[str] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        tree = _tree(path)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if LEGACY_DISPATCHERS & definitions.keys():
            failures.append(f"{relative}: retains a legacy dispatcher name")
        locals_enum = definitions.get("Locals")
        if not isinstance(locals_enum, ast.ClassDef) or not any(
            isinstance(base, ast.Name) and base.id == "Enum" for base in locals_enum.bases
        ):
            failures.append(f"{relative}: expected Locals enum")
        dispatcher = definitions.get("run_local")
        if not isinstance(dispatcher, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.append(f"{relative}: expected run_local dispatcher")
        else:
            args = dispatcher.args.args
            annotation = args[0].annotation if args else None
            if (
                not args
                or args[0].arg != "local"
                or not isinstance(annotation, ast.Name)
                or annotation.id != "Locals"
            ):
                failures.append(f"{relative}: expected run_local(local: Locals)")
        main_guards = [node for node in tree.body if _is_main_guard(node)]
        if len(main_guards) != 1 or not _main_calls_selected_local(main_guards[0]):
            failures.append(f"{relative}: main guard must select one Locals case directly")

    assert not failures, "example dispatcher violations:\n" + "\n".join(failures)
