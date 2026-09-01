# vanilla-option-pricers

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

## When to use it — and when not

Use `vanilla-option-pricers` for forward-based vanilla prices, Greeks, and implied-volatility
fits inside option-chain processing, volatility-surface preprocessing, simulation
post-processing, or calibration objectives.

It is deliberately not a derivatives framework: it does not construct spots, forwards, discount
curves, calendars, or settlement conventions, and it does not price American, exotic, or
stochastic-volatility models. For pricing and calibration under stochastic volatility, use
[`stochvolmodels`](https://github.com/ArturSepp/StochVolModels); for portfolio-level analytics
and reporting, use [`qis`](https://github.com/ArturSepp/QuantInvestStrats).

## Installation

### PyPI Installation
```bash
pip install vanilla-option-pricers
```

### Upgrade to Latest Version
```bash
pip install --upgrade vanilla-option-pricers
```

## Requirements

### Core Dependencies
- `python >= 3.10`
- `numba >= 0.60.0`
- `numpy >= 2.0`

The two runtime dependencies are NumPy and Numba. There is no dependency on any higher-level
analytics package.

## Supported Option Types

VanillaOptionPricers supports the following option types (passed as string parameters):

| Option Type | String Code | Description |
|-------------|-------------|-------------|
| Call | `'C'` | Standard call option |
| Put | `'P'` | Standard put option |
| Inverse-workflow call branch | `'IC'` | Caller supplies the required market normalisation |
| Inverse-workflow put branch | `'IP'` | Caller supplies the required market normalisation |

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

For arrays, chains, parity checks, and the Bachelier convention, clone the repository and use the
authoritative deterministic workflow:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/VanillaOptionPricers/blob/main/notebooks/offline_quickstart_colab.ipynb)

Use the authoritative, deterministic
[pricing and IV script](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/examples/getting_started/pricing_and_iv.py).
It prices one aligned BSM
slice, recovers the input implied volatilities, checks put-call parity, demonstrates the absolute
Bachelier volatility convention, and reports cold and warm execution separately.

```bash
python examples/getting_started/pricing_and_iv.py
```

The [rendered quickstart](https://vanillaoptionpricers.readthedocs.io/en/stable/getting_started.html)
includes that source directly and explains its inputs and output. The script requires no market
data, network access, credentials, or optional dependencies.

The Colab entry point installs the latest release from public PyPI, reports its exact version and
import path, and runs the same mechanically checked workflow with no saved notebook outputs.

Continue with the [Bachelier convention](docs/bachelier_convention.md),
[package-root API](docs/api.md), and [package comparison](docs/package_comparison.md) guides.

## Execution model

The public API has distinct execution paths:

- scalar pricing kernels are Numba dispatchers;
- slice, grid, and chain helpers use compiled loops over aligned inputs; and
- five `*_vector` convenience wrappers use `numpy.vectorize`.

The first Numba call includes compilation time. Runtime depends on input shape, dtype, machine,
and whether the relevant signature has already been compiled; this README makes no universal
timing or superiority claim.

## Typical uses

The package is intended for:

- quantitative research using forward-based European vanilla prices and Greeks;
- option-chain and volatility-surface preprocessing;
- calibration or simulation post-processing that needs price/volatility inversion; and
- numerical experiments or teaching examples built around explicit model inputs.

## Ecosystem

This package is part of Artur Sepp's open-source Python stack for quantitative finance. The
[maintainer profile](https://github.com/ArturSepp) is the canonical ten-package catalogue.

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor models and factor covariance estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain normalisation, reconstruction, queries, and visualisation |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Money-weighted multi-factor alpha from private-asset cash flows |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic volatility pricing analytics |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) *(this package)* | Numba-vectorised BSM/Bachelier prices, Greeks, and implied-volatility fits |

`vanilla-option-pricers` is standalone and depends only on NumPy and Numba. It supplies the
vanilla pricing layer used by `option-chain-analytics` and `stochvolmodels`; those packages own
point-in-time chain workflows and stochastic-volatility models respectively.

## Feedback & contributing

- **Bug or unsupported edge case:** [open the bug-report form](https://github.com/ArturSepp/VanillaOptionPricers/issues/new?template=bug_report.yml)
  with the package/Python versions, platform, quote convention, inputs, expected result, and actual
  result.
- **Feature:** [open the feature-request form](https://github.com/ArturSepp/VanillaOptionPricers/issues/new?template=feature_request.yml)
  with the payoff, quote convention, or model edge case that is unsupported, your current
  workaround, and the smallest useful API.
- **Contribution:** read [`CONTRIBUTING.md`](CONTRIBUTING.md), then browse
  [`good first issue`](https://github.com/ArturSepp/VanillaOptionPricers/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
  or [`help wanted`](https://github.com/ArturSepp/VanillaOptionPricers/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
  work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

If you use VanillaOptionPricers in your research, please cite it as:

```bibtex
@software{sepp2026vanillaoptionpricers,
  title={VanillaOptionPricers: Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks, and implied-volatility fits over NumPy arrays},
  author={Sepp, Artur},
  year={2026},
  version={2.1.0},
  url={https://github.com/ArturSepp/VanillaOptionPricers},
  note={Python package for forward-based vanilla option pricing and implied-volatility fitting}
}
```
