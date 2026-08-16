# Examples

Examples import the installed public `vanilla_option_pricers` API. From the repository root,
install the project before running them:

```bash
pip install -e ".[dev]"
```

## Getting started

Deterministic, offline-first examples will live in `getting_started/`. They are intended for
new users and must require no credentials, downloads, or optional data dependencies.

## Performance diagnostics

Scripts in `performance/` are manual timing diagnostics for maintainers. They are informative,
machine-dependent, and are not correctness or CI gates. Run the current diagnostic with:

```bash
python examples/performance/bsm_speed.py
```

The root `examples/` directory is repository-only and is not included in the wheel.
