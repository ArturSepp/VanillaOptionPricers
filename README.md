# VanillaOptionPricers (`vanilla-option-pricers`)

**`vanilla-option-pricers` - Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks,
and implied-volatility fits over NumPy arrays for quantitative research pipelines.**

Install the distribution as `vanilla-option-pricers` and import it as
`vanilla_option_pricers`.

[![PyPI](https://img.shields.io/pypi/v/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![Python](https://img.shields.io/pypi/pyversions/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![License](https://img.shields.io/github/license/ArturSepp/VanillaOptionPricers.svg?style=flat-square)](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/LICENSE.txt)
[![CI](https://github.com/ArturSepp/VanillaOptionPricers/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/VanillaOptionPricers/actions)
[![Downloads](https://static.pepy.tech/badge/vanilla-option-pricers)](https://pepy.tech/project/vanilla-option-pricers)
[![Monthly](https://static.pepy.tech/badge/vanilla-option-pricers/month)](https://pepy.tech/project/vanilla-option-pricers)

## Why vanilla-option-pricers

Research pipelines often need focused pricing functions rather than a derivatives framework.
This package provides forward-based Black-Scholes-Merton and Bachelier prices, Greeks, and
implied-volatility inversion through scalar functions, aligned-array helpers, and per-expiry
containers. The runtime dependency surface is limited to NumPy and Numba.

## What makes it different

- **Log-normal and relative-normal models side by side.** Black-Scholes-Merton and Bachelier
  functions share a forward-and-discount-factor interface. The detailed Bachelier volatility
  convention is part of the package contract and will be documented separately.
- **Implied volatility as a first-class fitter.** Scalar, slice, and chain helpers recover model
  volatilities from caller-supplied option prices.
- **Inverse-workflow branches.** The `'IC'` and `'IP'` codes select branches used in
  coin-denominated inverse-option workflows. They do not perform every quote, numeraire, or payoff
  normalization required by a market contract; callers remain responsible for those conversions.
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
| Inverse-workflow call branch | `'IC'` | Caller supplies the required market normalization |
| Inverse-workflow put branch | `'IP'` | Caller supplies the required market normalization |

## Quick Start

### Basic Option Pricing

Pricers consume a caller-supplied `forward` and `discfactor`; the package does not construct them
from spots, rates, dividends, or curves. Time to maturity is expressed in years and `vol` is an
annualised log-normal volatility in the Black-Scholes-Merton functions.

```python
from vanilla_option_pricers import (
    compute_bsm_vanilla_price,
    compute_bsm_vanilla_delta,
    infer_bsm_implied_vol,
)

forward = 101.25         # supplied by the caller
discfactor = 0.99        # supplied by the caller
strike = 105.0
ttm = 0.25             # time to maturity, in years
vol = 0.20             # annualised lognormal vol

price = compute_bsm_vanilla_price(
    forward=forward,
    strike=strike,
    ttm=ttm,
    vol=vol,
    optiontype='C',
    discfactor=discfactor,
)
delta = compute_bsm_vanilla_delta(
    ttm=ttm,
    forward=forward,
    strike=strike,
    vol=vol,
    optiontype='C',
    discfactor=discfactor,
)

# invert the price back to an implied vol (round-trips to `vol`)
implied_vol = infer_bsm_implied_vol(
    forward=forward,
    ttm=ttm,
    strike=strike,
    given_price=price,
    discfactor=discfactor,
    optiontype='C',
)

print(f"price={price:.4f}  delta={delta:.4f}  implied_vol={implied_vol:.4f}")
```

### Vectorised calculations

```python
import numpy as np
from vanilla_option_pricers import compute_bsm_vanilla_price_vector

# Vectorised pricing for multiple strikes
forwards = np.array([95, 100, 105, 110])
strikes = np.array([100, 100, 100, 100])
vols = np.array([0.15, 0.20, 0.25, 0.30])

option_prices = compute_bsm_vanilla_price_vector(
    forward=forwards,
    strike=strikes,
    ttm=0.25,
    vol=vols,
    optiontype='C'
)

print("Vectorised option prices:", option_prices)
```


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

This package is part of an open-source Python stack for quantitative finance — full catalogue at [github.com/ArturSepp](https://github.com/ArturSepp):

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor models and factor covariance estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic volatility pricing analytics |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) *(this package)* | Numba-vectorised BSM/Bachelier prices, Greeks, and implied-volatility fits |

Dependency links within the stack: `optimalportfolios` builds on `qis` and `factorlasso`; `trendfollowing` builds on `qis`.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
git clone https://github.com/ArturSepp/VanillaOptionPricers.git
cd VanillaOptionPricers
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Citation

If you use VanillaOptionPricers in your research, please cite it as:

```bibtex
@software{sepp2024vanillaoptionpricers,
  title={VanillaOptionPricers: Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks, and implied-volatility fits over NumPy arrays},
  author={Sepp, Artur},
  year={2024},
  url={https://github.com/ArturSepp/VanillaOptionPricers},
  note={Python package for forward-based vanilla option pricing and implied-volatility fitting}
}
```
