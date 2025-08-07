# VanillaOptionPricers

[![PyPI Version](https://img.shields.io/pypi/v/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/vanilla-option-pricers?style=flat-square)](https://pypi.org/project/vanilla-option-pricers/)
[![License](https://img.shields.io/github/license/ArturSepp/VanillaOptionPricers.svg?style=flat-square)](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ArturSepp/VanillaOptionPricers?style=flat-square&logo=github)](https://github.com/ArturSepp/VanillaOptionPricers)
[![GitHub Forks](https://img.shields.io/github/forks/ArturSepp/VanillaOptionPricers?style=flat-square&logo=github)](https://github.com/ArturSepp/VanillaOptionPricers)
[![Monthly Downloads](https://pepy.tech/badge/vanilla-option-pricers/month)](https://pepy.tech/project/vanilla-option-pricers)
[![Weekly Downloads](https://pepy.tech/badge/vanilla-option-pricers/week)](https://pepy.tech/project/vanilla-option-pricers)

**Fast and vectorized option pricers and implied volatility fitters for Black-Scholes and Merton models**

VanillaOptionPricers is a high-performance Python package that provides fast, vectorized implementations of option pricing models and implied volatility calculations. Built with Numba for optimal performance, this package is designed for quantitative analysts, traders, and researchers who need efficient option pricing capabilities.

## Key Features

- **High Performance**: Vectorized implementations using Numba for maximum speed
- **Multiple Models**: Support for Black-Scholes-Merton and Bachelier normal models
- **Implied Volatility**: Fast and robust implied volatility solvers
- **Option Types**: Support for vanilla calls, puts, and inverse options
- **Minimal Dependencies**: Lightweight with core dependencies on NumPy and Numba only
- **Easy Integration**: Simple API for seamless integration into existing workflows

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
- `python >= 3.8, < 3.11`
- `numba >= 0.59.0`
- `numpy >= 1.26.4`

The package maintains minimal dependencies on higher-level packages, ensuring fast installation and reduced compatibility issues.

## Supported Option Types

VanillaOptionPricers supports the following option types (passed as string parameters):

| Option Type | String Code | Description |
|-------------|-------------|-------------|
| Call | `'C'` | Standard call option |
| Put | `'P'` | Standard put option |
| Inverse Call | `'IC'` | Inverse call option |
| Inverse Put | `'IP'` | Inverse put option |

## Quick Start

### Basic Option Pricing

```python
import numpy as np
from vanilla_option_pricers import black_scholes_price, implied_volatility

# Define option parameters
spot = 100.0          # Current underlying price
strike = 105.0        # Strike price
time_to_expiry = 0.25 # Time to expiration (in years)
risk_free_rate = 0.05 # Risk-free interest rate
volatility = 0.20     # Volatility
option_type = 'C'     # Call option

# Calculate option price
option_price = black_scholes_price(
    spot=spot,
    strike=strike,
    time_to_expiry=time_to_expiry,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    option_type=option_type
)

print(f"Option Price: ${option_price:.4f}")
```

### Vectorized Calculations

```python
import numpy as np
from vanilla_option_pricers import compute_bsm_vanilla_price_vector

# Vectorized pricing for multiple strikes
forwards = np.array([95, 100, 105, 110])
strikes = np.array([100, 100, 100, 100])
vols = np.array([0.15, 0.20, 0.25, 0.30])

option_prices = compute_bsm_vanilla_price_vector(
    forward=forwards,
    strike=strikes,
    ttm=0.25,
    vol=vols,
    option_type='C'
)

print("Vectorized Option Prices:", option_prices)
```


## Performance Benefits

VanillaOptionPricers leverages Numba's JIT compilation to achieve:

- **Vectorization**: Process arrays of parameters efficiently
- **Speed**: Orders of magnitude faster than pure Python implementations
- **Memory Efficiency**: Optimized memory usage for large-scale calculations
- **Numerical Stability**: Robust implementations with proper handling of edge cases

## API Documentation

### Core Functions

#### `black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)`

Calculate Black-Scholes option price.

**Parameters:**
- `spot` (float/array): Current underlying price
- `strike` (float/array): Strike price
- `time_to_expiry` (float/array): Time to expiration in years
- `risk_free_rate` (float/array): Risk-free interest rate
- `volatility` (float/array): Volatility (annualized)
- `option_type` (str): Option type ('C', 'P', 'IC', 'IP')

**Returns:**
- `float/array`: Option price(s)

#### `implied_volatility(market_price, spot, strike, time_to_expiry, risk_free_rate, option_type)`

Calculate implied volatility from market price.

**Parameters:**
- `market_price` (float/array): Observed market price
- `spot` (float/array): Current underlying price
- `strike` (float/array): Strike price
- `time_to_expiry` (float/array): Time to expiration in years
- `risk_free_rate` (float/array): Risk-free interest rate
- `option_type` (str): Option type ('C', 'P', 'IC', 'IP')

**Returns:**
- `float/array`: Implied volatility(ies)

## Use Cases

VanillaOptionPricers is ideal for:

- **Quantitative Research**: Academic research requiring fast option pricing
- **Trading Systems**: Real-time option pricing in trading applications
- **Risk Management**: Portfolio risk calculations and scenario analysis
- **Market Making**: High-frequency option pricing and implied volatility calculations
- **Financial Education**: Teaching option pricing concepts with efficient implementations

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
git clone https://github.com/ArturSepp/VanillaOptionPricers.git
cd VanillaOptionPricers
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

***BibTeX Citation***

If you use VanillaOptionPricers in your research, please cite it as:

```bibtex
@software{sepp2024vanillaoptionpricers,
  title={VanillaOptionPricers: Fast and vectorized option pricers and implied volatility fitters for Black-Scholes and Merton models},
  author={Sepp, Artur},
  year={2024},
  url={https://github.com/ArturSepp/VanillaOptionPricers},
  note={Python package for high-performance option pricing}
}
```
