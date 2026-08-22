# Contributing to VanillaOptionPricers

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
