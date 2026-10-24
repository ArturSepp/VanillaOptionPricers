# vanilla-option-pricers

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2024-06-08](https://github.com/ArturSepp/VanillaOptionPricers/commit/31caec51af9d4ef61d4f40b5b0d88303c49d950d)*

Implemented in [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers).
Software citation: [CITATION.cff](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff).

**Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks, and implied-volatility fits
over NumPy arrays for quantitative research pipelines**

It is a focused, function-based numerical library, not a derivatives framework: callers supply
forwards, discount factors, maturities, and market conventions.

**Install:** `pip install vanilla-option-pricers` · **Import:** `vanilla_option_pricers` · **Status:** Beta

[![PyPI](https://img.shields.io/pypi/v/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![Python](https://img.shields.io/pypi/pyversions/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![CI](https://github.com/ArturSepp/VanillaOptionPricers/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/VanillaOptionPricers/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/vanillaoptionpricers/badge/?version=latest)](https://vanillaoptionpricers.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/VanillaOptionPricers.svg?style=flat-square)](LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/vanilla-option-pricers)](https://pepy.tech/project/vanilla-option-pricers)
[![Monthly](https://static.pepy.tech/badge/vanilla-option-pricers/month)](https://pepy.tech/project/vanilla-option-pricers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/VanillaOptionPricers/blob/main/notebooks/offline_quickstart_colab.ipynb)

**Documentation:** [vanillaoptionpricers.readthedocs.io](https://vanillaoptionpricers.readthedocs.io/en/latest/) ·
[package-root API](docs/api.md) · [Bachelier convention](docs/bachelier_convention.md) ·
[package comparison](docs/package_comparison.md)

---

## Why vanilla-option-pricers

Research pipelines often need focused pricing functions rather than a derivatives framework.
This package provides forward-based Black-Scholes-Merton and Bachelier prices, Greeks, and
implied-volatility inversion through scalar functions, aligned-array helpers, and per-expiry
containers. The runtime dependency surface is limited to NumPy and Numba.

### Key differentiators

- **Log-normal and absolute-normal models side by side.** Black-Scholes-Merton and Bachelier
  functions share a forward-and-discount-factor interface. The
  [Bachelier convention](docs/bachelier_convention.md) is annualised absolute normal volatility
  in the same units as the forward and strike.
- **Implied volatility as a first-class fitter.** Scalar, slice, and chain helpers recover model
  volatilities from caller-supplied option prices.
- **Inverse-workflow branches.** The `'IC'` and `'IP'` codes select branches used in
  coin-denominated inverse-option workflows. They do not perform every quote, numeraire, or payoff
  normalisation required by a market contract; callers remain responsible for those conversions.
  For the contract theory, see Lucic and Sepp (2024), *Valuation and Hedging of Cryptocurrency
  Inverse Options*, Quantitative Finance, 24(7), 851–869.
- **Two runtime dependencies.** NumPy and Numba; no object hierarchy, calendar, curve, pandas, or
  SciPy layer.

### When to use it — and when not

Use `vanilla-option-pricers` for forward-based vanilla prices, Greeks, and implied-volatility
fits inside option-chain processing, volatility-surface preprocessing, simulation
post-processing, calibration objectives, and numerical experiments or teaching examples built
around explicit model inputs.

It is deliberately not a derivatives framework: it does not construct spots, forwards, discount
curves, calendars, or settlement conventions, and it does not price American, exotic, or
stochastic-volatility models. For pricing and calibration under stochastic volatility, use
[`stochvolmodels`](https://github.com/ArturSepp/StochVolModels); for portfolio-level analytics
and reporting, use [`qis`](https://github.com/ArturSepp/QuantInvestStrats).

## Installation

```bash
pip install vanilla-option-pricers
```

Upgrade using

```bash
pip install --upgrade vanilla-option-pricers
```

Core dependencies:
- `python >= 3.10`
- `numba >= 0.60.0`
- `numpy >= 2.0`

The two runtime dependencies are NumPy and Numba. There is no dependency on any higher-level
analytics package.

## Table of contents

1. [Why vanilla-option-pricers](#why-vanilla-option-pricers)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Citation](#citation)
11. [Ecosystem](#ecosystem)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Statement of need

Every option-chain, volatility-surface and stochastic-volatility workflow in the stack needs the
same small set of vanilla operations — a forward-based price, its Greeks, and the implied
volatility that recovers a quote — applied to arrays of strikes and expiries without a
derivatives framework around them. `vanilla-option-pricers` is that layer: it supplies the
vanilla pricing used by `option-chain-analytics` and `stochvolmodels`, with the Bachelier
convention stated explicitly and the inverse-option branches of Lucic and Sepp (2024) available
for coin-denominated workflows. No research paper is implemented in this package beyond those
branches.

| Paper | Implemented analytics | Where |
|---|---|---|
| Lucic, V. and Sepp, A. (2024), *Valuation and Hedging of Cryptocurrency Inverse Options*, Quantitative Finance 24(7), 851–869 | The `'IC'` and `'IP'` inverse-workflow branches of the pricing and implied-volatility functions; quote, numeraire and payoff normalisation remain with the caller | [inverse options](docs/inverse_options.md) |

## Overview

The public import surface in `src/vanilla_option_pricers/__init__.py` is inventoried, with
signatures, in the [package-root API](docs/api.md) page. The functions fall into three execution
tiers: scalar pricing kernels are Numba dispatchers; slice, grid, and chain helpers use compiled
loops over aligned inputs; and five `*_vector` convenience wrappers use `numpy.vectorize`. The
first Numba call includes compilation time. Runtime depends on input shape, dtype, machine, and
whether the relevant signature has already been compiled; this README makes no universal timing
or superiority claim. See [array shapes and Numba](docs/array_shapes_and_numba.md) and
[numerical accuracy and performance](docs/numerical_accuracy_and_performance.md).

Supported option types, passed as string parameters:

| Option Type | String Code | Description |
|-------------|-------------|-------------|
| Call | `'C'` | Standard call option |
| Put | `'P'` | Standard put option |
| Inverse-workflow call branch | `'IC'` | Caller supplies the required market normalisation |
| Inverse-workflow put branch | `'IP'` | Caller supplies the required market normalisation |

### Analytics at a glance

| Area | What the package computes |
|---|---|
| Black-Scholes-Merton | Forward-based vanilla prices and Greeks from forward, strike, maturity, volatility, option type and discount factor (`compute_bsm_vanilla_price` and the aligned-array and chain helpers); see [pricing and Greeks](docs/pricing_and_greeks.md) |
| Bachelier | Absolute-normal prices, Greeks and forward-delta strike conversion (`compute_normal_price`, `compute_normal_slice_prices`) in annualised absolute normal volatility, the same units as the forward and strike, including negative forwards; see the [Bachelier convention](docs/bachelier_convention.md) |
| Implied volatility | Scalar, slice and chain inversion of caller-supplied prices (`infer_bsm_implied_vol`, `infer_normal_implied_vol`); see [implied volatility](docs/implied_volatility.md) |
| Inverse-option branches | `'IC'`/`'IP'` branches for coin-denominated workflows; see [inverse options](docs/inverse_options.md) |
| Execution tiers | Scalar Numba kernels, compiled slice/grid/chain loops, and five `numpy.vectorize` wrappers |

## Five-minute quickstart

This scalar price-to-implied-volatility round trip works from any directory after installation:

```python
from vanilla_option_pricers import compute_bsm_vanilla_price, infer_bsm_implied_vol

forward, discfactor, ttm, strike, vol = 101.25, 0.99, 0.25, 105.0, 0.20
price = compute_bsm_vanilla_price(forward, strike, ttm, vol, "C", discfactor)
implied_vol = infer_bsm_implied_vol(
    forward, ttm, strike, price, discfactor, "C"
)
print(f"call_price={price:.6f} implied_vol={implied_vol:.8f}")
```

```text
call_price=2.481051 implied_vol=0.20000000
```

## Examples

### Highlighted examples

For arrays, chains, parity checks, and the Bachelier convention, clone the repository and use the
authoritative deterministic
[pricing and IV script](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/examples/getting_started/pricing_and_iv.py).
It prices one aligned BSM slice, recovers the input implied volatilities, checks put-call parity,
demonstrates the absolute Bachelier volatility convention, and reports cold and warm execution
separately:

```bash
python examples/getting_started/pricing_and_iv.py
```

The [rendered quickstart](https://vanillaoptionpricers.readthedocs.io/en/stable/getting_started.html)
includes that source directly and explains its inputs and output. The script requires no market
data, network access, credentials, or optional dependencies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/VanillaOptionPricers/blob/main/notebooks/offline_quickstart_colab.ipynb)

The Colab entry point installs the latest release from public PyPI, reports its exact version and
import path, and runs the same mechanically checked workflow with no saved notebook outputs.

### Runnable examples

Deterministic, offline-first examples live in [`examples/`](examples/) (see
[`examples/README.md`](examples/README.md)):

- [`getting_started/pricing_and_iv.py`](examples/getting_started/pricing_and_iv.py) — the
  shortest BSM/Bachelier first-success workflow.
- [`getting_started/normal_vol_workflow.py`](examples/getting_started/normal_vol_workflow.py) —
  absolute-normal units, price-to-IV recovery, forward-delta strike conversion, slice prices,
  deltas, vegas, and a negative-forward rate example.
- [`performance/bsm_speed.py`](examples/performance/bsm_speed.py) — how fast one ordinary
  Black-Scholes-Merton call chain is priced on the running machine.

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/vanilla_option_pricers/ examples/
```

The suite needs no market data, network access or optional dependencies. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the development commands and numerical-change rules.

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug or unsupported edge case:** [open the bug-report form](https://github.com/ArturSepp/VanillaOptionPricers/issues/new?template=bug_report.yml)
  with the package/Python versions, platform, quote convention, inputs, expected result, and actual
  result.
- **Feature:** [open the feature-request form](https://github.com/ArturSepp/VanillaOptionPricers/issues/new?template=feature_request.yml)
  with the payoff, quote convention, or model edge case that is unsupported, your current
  workaround, and the smallest useful API.
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/VanillaOptionPricers/issues)
  and name the function, quote convention, or model involved.
- **Contribution:** read [`CONTRIBUTING.md`](CONTRIBUTING.md), then browse
  [`good first issue`](https://github.com/ArturSepp/VanillaOptionPricers/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
  or [`help wanted`](https://github.com/ArturSepp/VanillaOptionPricers/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
  work.

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

If you use VanillaOptionPricers in your research, please cite it as:

```bibtex
@software{sepp2026vanillaoptionpricers,
  title={VanillaOptionPricers: Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks, and implied-volatility fits over NumPy arrays},
  author={Sepp, Artur},
  year={2026},
  version={2.2.1},
  url={https://github.com/ArturSepp/VanillaOptionPricers},
  note={Python package for forward-based vanilla option pricing and implied-volatility fitting}
}
```

```bibtex
@article{lucicsepp2024,
  title={Valuation and hedging of cryptocurrency inverse options},
  author={Lucic, Vladimir and Sepp, Artur},
  journal={Quantitative Finance},
  volume={24},
  number={7},
  pages={851--869},
  year={2024}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) *(this package)* | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

`vanilla-option-pricers` is standalone and depends only on NumPy and Numba. It supplies the
vanilla pricing layer used by `option-chain-analytics` and `stochvolmodels`; those packages own
point-in-time chain workflows and stochastic-volatility models respectively.

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Disclaimer

`vanilla-option-pricers` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE.txt](LICENSE.txt) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
