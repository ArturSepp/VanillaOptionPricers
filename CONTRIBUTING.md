# Contributing to VanillaOptionPricers

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-22](https://github.com/ArturSepp/VanillaOptionPricers/commit/471989cc2b140f9bec00c9f7440b5f8eda162ddf)*

Source: [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers).
Software citation: [CITATION.cff](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff).

Thanks for your interest in `vanilla-option-pricers`. The package is deliberately small: it
provides fast, function-based Black-Scholes-Merton and Bachelier analytics without becoming a
general derivatives framework.

## Scope

In scope:

- Bug fixes in vanilla pricing, Greeks, array helpers, or implied-volatility inversion
- Numerical robustness improvements backed by an independent reference or parity check
- Compatibility work for supported Python, NumPy, and Numba releases
- Documentation, deterministic examples, packaging, and tests

Open an issue before writing code that changes numerical conventions, public signatures,
dependency floors, or runtime dependencies. American, exotic, and stochastic-volatility models
belong elsewhere; portfolio analytics and reporting belong in the sibling packages documented in
`AGENTS.md`. Do not submit proprietary data, generated output, or local environment files.

## Reporting a bug

Use the bug-report template and include the package version, Python version, operating system, a
minimal self-contained reproducer, and the full traceback or incorrect output. For a numerical
problem, state the forward, strike, expiry, discount factor, volatility convention, and option
type, and include the independent value or identity used for comparison.

## Starter tasks

These are bounded proposals, not assigned issues or confirmed defects. Recheck current source
and open work, then agree one task with the maintainer. Keep numerical kernels, public
signatures, dependencies and example/notebook source unchanged for these documentation tasks.
Use the existing [feature form](https://github.com/ArturSepp/VanillaOptionPricers/issues/new?template=feature_request.yml)
with Area **Documentation or deterministic example**. Include the task ID, exact paths,
reproduction, acceptance and focused check under Proposed behavior; retain both boundary
checkboxes. Use the bug form only for observed incorrect behaviour.

### VOP-1: clarify first-use output

Owner: VanillaOptionPricers. Edit only `docs/getting_started.md` to clarify one observed setup
or output-reading ambiguity. Preserve the canonical script, literalinclude and checked Colab
mirror. Acceptance: another reader can locate the four prices and both error checks.

```console
python examples/getting_started/pricing_and_iv.py
python -m pytest src/vanilla_option_pricers/tests/test_colab_quickstart.py -q
```

### VOP-2: clarify Bachelier units

Owner: VanillaOptionPricers. Edit only `docs/bachelier_convention.md` to clarify one confusion
between absolute normal volatility and lognormal percentage volatility. Preserve formulas,
constants and results. Acceptance: the forward, strike and normal-volatility units are explicit.

```console
python -m pytest src/vanilla_option_pricers/tests/test_bachelier.py -k "absolute or negative" -q
```

### VOP-3: check release navigation

Owner: VanillaOptionPricers. Edit only `docs/getting_started.md` to clarify one stable/latest
or source-only boundary, linking the same canonical example. Acceptance: a reader can distinguish
the tested released path from main. Preserve models, release metadata and the theme.

```console
python -m pytest src/vanilla_option_pricers/tests/test_docs_site.py -q
```

Run the package's existing strict HTML command after a documentation edit and inspect the
rendered passage. On the maintainer OneDrive checkout, first dot-source the shared
`Enter-AgentRepo.ps1` and use `C:\Python\VanillaOptionPricers312\Scripts\python.exe`;
keep caches, builds and disposable source exports on C. The normal `uv` commands below are for
ordinary contributor checkouts outside OneDrive; do not create `.venv` in a synchronized folder.

For review, follow the [shared contribution review guidance](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md#review-attribution-and-automation-assisted-changes).
State source revision, actual versions, exact checks, skipped checks, rendered review and material
automation assistance. Keep the MIT boundary and existing attribution. Automated command runs
do not substitute for an independent reader; no response-time or acceptance promise is made.

## Development setup

```bash
git clone https://github.com/ArturSepp/VanillaOptionPricers.git
cd VanillaOptionPricers
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/vanilla_option_pricers/ examples/
```

Build the documentation with the same warning gate used in CI:

```bash
uv sync --locked --extra docs
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs docs/_build/html
uv run --no-sync python -m sphinx -E -W -b linkcheck docs docs/_build/linkcheck
```

`--locked` intentionally fails when `pyproject.toml` and `uv.lock` disagree. Dependency groups are
contributor environments, not package extras: use `test` for pytest, `lint` for Ruff, and retain
the `docs` extra for Read the Docs.

## Pull requests

- Keep one focused topic per pull request.
- Add a regression test that fails before a behavioral fix and passes afterwards.
- Preserve explicit forward, discount-factor, expiry, payoff, and volatility conventions.
- Verify numerical changes against an independent reference value or put-call parity.
- Keep the public API function-based and the runtime dependency surface to NumPy and Numba.
- Do not commit generated output, proprietary data, local paths, or development environments.
- Run the relevant test, lint, documentation, and wheel checks before submitting.
- Do not bump package or citation versions; releases are handled separately.

## Conduct and licence

Be civil and assume good faith. By contributing, you agree that your contribution is licensed
under the project's MIT licence.
