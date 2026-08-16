---
myst:
  html_meta:
    description: >-
      vanilla-option-pricers provides Numba-vectorised Black-Scholes-Merton and Bachelier prices,
      Greeks, and implied-volatility fits over NumPy arrays for quantitative research pipelines.
---

# vanilla-option-pricers

`vanilla-option-pricers` provides Numba-vectorised Black-Scholes-Merton and Bachelier prices,
Greeks, and implied-volatility fits over NumPy arrays for quantitative research pipelines.

Install the distribution as `vanilla-option-pricers` and import it as
`vanilla_option_pricers`. The public functions consume caller-supplied forwards, discount
factors, maturities, strikes, volatilities, option prices, and option-type codes.

Start with [installation and a first result](getting_started.md). The deterministic example
prices a call, computes its delta, and recovers its input Black-Scholes-Merton volatility without
network access or credentials.

Continue with the task guide that matches your workflow:

- [Price options and compute Greeks](pricing_and_greeks.md)
- [Recover implied volatility](implied_volatility.md)
- [Look up the complete package-root API](api.md)

Use the convention and trust guides before adapting market data or measuring performance:

- [Convert Bachelier normal-volatility units](bachelier_convention.md)
- [Normalize inverse-option quotes](inverse_options.md)
- [Choose array shapes and understand Numba behavior](array_shapes_and_numba.md)
- [Interpret numerical accuracy and timing](numerical_accuracy_and_performance.md)

## Scope

The package covers:

- forward-based Black-Scholes-Merton and Bachelier vanilla pricing;
- selected Greeks;
- scalar, aligned-array, grid, and per-expiry execution paths; and
- implied-volatility fitting from caller-supplied option prices.

It does not construct spots, forwards, discount curves, calendars, or settlement conventions. It
does not price American, exotic, path-dependent, or stochastic-volatility models. The `IC` and
`IP` codes are inverse-workflow branches; callers own the required quote, numeraire, and payoff
normalisation.

## Project resources

- [PyPI package](https://pypi.org/project/vanilla-option-pricers/)
- [Source repository](https://github.com/ArturSepp/VanillaOptionPricers)
- [Issue tracker](https://github.com/ArturSepp/VanillaOptionPricers/issues)
- [Changelog](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff)

```{toctree}
:maxdepth: 2
:caption: Documentation

getting_started
pricing_and_greeks
implied_volatility
api
bachelier_convention
inverse_options
array_shapes_and_numba
numerical_accuracy_and_performance
```
