# Installation and a first result

## Install

Install the released package from PyPI:

```console
python -m pip install vanilla-option-pricers
```

For a source checkout, install the package and the build-only documentation tools separately:

```console
python -m pip install -e ".[docs]"
```

The package's runtime dependencies remain NumPy and Numba. Sphinx, MyST Parser, and Furo belong
only to the optional `docs` extra.

## Price, compute delta, and recover implied volatility

This example uses the public package-root API. The caller supplies the forward and discount
factor; the package does not construct them from a spot, rate, dividend, or curve.

```python
from vanilla_option_pricers import (
    compute_bsm_vanilla_delta,
    compute_bsm_vanilla_price,
    infer_bsm_implied_vol,
)

forward = 101.25
discfactor = 0.99
strike = 105.0
ttm = 0.25
vol = 0.20

price = compute_bsm_vanilla_price(
    forward=forward,
    strike=strike,
    ttm=ttm,
    vol=vol,
    optiontype="C",
    discfactor=discfactor,
)
delta = compute_bsm_vanilla_delta(
    ttm=ttm,
    forward=forward,
    strike=strike,
    vol=vol,
    optiontype="C",
    discfactor=discfactor,
)
implied_vol = infer_bsm_implied_vol(
    forward=forward,
    ttm=ttm,
    strike=strike,
    given_price=price,
    discfactor=discfactor,
    optiontype="C",
)

print(f"price={price:.4f}  delta={delta:.4f}  implied_vol={implied_vol:.4f}")
```

Expected output:

```text
price=2.4811  delta=0.3731  implied_vol=0.2000
```

The implied-volatility result round-trips to the input annualised lognormal volatility. Time to
maturity is in years. The exact same example is checked in the repository
[README quick start](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/README.md#quick-start).

## Next steps and boundaries

Use the package-root functions for stable user examples. Array helpers have distinct scalar,
aligned-array, grid, and per-expiry contracts; arbitrary broadcasting is not promised. The first
call to a Numba-compiled signature includes compilation time.

Detailed task guides and a complete API inventory follow in later documentation stages. Until
then, use the [source repository](https://github.com/ArturSepp/VanillaOptionPricers) and
[issue tracker](https://github.com/ArturSepp/VanillaOptionPricers/issues) for implementation and
support details.
