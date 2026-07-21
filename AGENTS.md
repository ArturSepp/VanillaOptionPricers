# AGENTS.md

Guidance for AI coding agents working in the **VanillaOptionPricers** repository.

## Project overview

`vanilla-option-pricers` provides fast, vectorised pricers and implied volatility
fitters for vanilla options under the Black-Scholes-Merton and Bachelier (normal)
models, including coin-denominated inverse options as traded on cryptocurrency
derivatives exchanges. Everything is `numba`-compiled over numpy arrays, with exactly
two runtime dependencies: numpy and numba.

The design goal is minimalism and speed inside calibration loops and surface fitters —
it is deliberately not a derivatives framework. Distribution name
`vanilla-option-pricers`; import name `vanilla_option_pricers`. Licensed MIT
(`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
vanilla_option_pricers/
  black_scholes.py   Black-Scholes-Merton pricers and implied volatility fitters
  bachelier.py       Bachelier normal pricers and implied volatility fitters
  utils.py           shared numerical helpers
  tests/
    bsm_speed.py     performance check, run directly by CI
```

## Commands

```bash
pip install -e ".[dev]"
python vanilla_option_pricers/tests/bsm_speed.py   # performance check, as CI runs it
ruff check vanilla_option_pricers/                 # lint
```

There is no pytest configuration in this repository yet and no `test_*.py` modules;
CI runs the speed script directly. Supported Python is >= 3.9; CI runs 3.10 - 3.12.

## Conventions

- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`).
- Every pricing function is `numba`-compiled and takes numpy arrays in and returns
  numpy arrays out. There is no object model, no calendar handling, and no pandas.
- Option type selection is by enum, including the inverse-option variants.
- Scalar inputs are handled by broadcasting, not by a separate scalar code path.
- New functionality should come with a numerical check against a reference value or
  against put-call parity.

## Constraints — do not do these

- Do not add runtime dependencies. numpy and numba only — in particular, do not import
  `scipy` (for example for root finding in the implied volatility fitters) or `pandas`.
  The two-dependency footprint is the reason this package exists.
- Do not add American, exotic, or path-dependent payoffs, term structures, settlement
  conventions, or stochastic volatility. Those belong in `stochvolmodels`.
- Do not introduce Python-level loops over strikes or expiries in place of vectorised
  operations.
- Do not wrap the functions in classes; the API is deliberately function-based.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

`pyproject.toml` is at version 1.2.3 while PyPI serves 1.2.2, so the repository is one
unpublished version ahead. Adding a real pytest suite (currently only `bsm_speed.py`
exists, run directly by CI) would be a welcome contribution if the maintainer asks.
