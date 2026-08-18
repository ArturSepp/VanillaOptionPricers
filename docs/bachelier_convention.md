---
myst:
  html_meta:
    description: >-
      Use annualised absolute Bachelier normal volatility correctly with
      vanilla-option-pricers.
---

# Bachelier normal-volatility convention

Use this page before passing a normal volatility to `compute_normal_price` or an
`infer_normal_*` function.

```{important}
`vol` is annualised **absolute normal volatility**. It has the same units as `forward`
and `strike`; it is not a dimensionless lognormal or relative-normal volatility.
```

The package defines

$$
sdev = vol\sqrt{T},
$$

where `vol` is the annualised absolute normal volatility in price units per square-root
year and `sdev` is the standard deviation over the option's maturity.

## Convert an external quote

| Starting quote | Package input |
|---|---|
| Absolute normal volatility $\sigma_N$ | `vol = sigma_n` |
| Maturity standard deviation `sdev` | `vol = sdev / sqrt(ttm)` |
| Relative normal volatility $\sigma_N/F$ | `vol = relative_vol * forward` |

The package does not infer units or inspect vendor metadata. A quote of `150` basis points
on a rate expressed as `0.04` is `vol=0.015`; a normal volatility of `5.0` on an equity
forward is `vol=5.0`.

## Executed reference checks

```python
from vanilla_option_pricers import (
    compute_normal_delta,
    compute_normal_price,
    infer_normal_implied_vol,
)

forward = 100.0
strike = 102.0
ttm = 0.50
discfactor = 0.98
vol = 5.0

call = compute_normal_price(forward, strike, ttm, vol, discfactor, "C")
put = compute_normal_price(forward, strike, ttm, vol, discfactor, "P")
delta = compute_normal_delta(ttm, forward, strike, vol, "C", discfactor)

h = 1e-4
price_up = compute_normal_price(forward + h, strike, ttm, vol, discfactor, "C")
price_down = compute_normal_price(forward - h, strike, ttm, vol, discfactor, "C")
delta_fd = (price_up - price_down) / (2.0 * h)

implied = infer_normal_implied_vol(
    forward, ttm, strike, call, discfactor, "C"
)

print(f"absolute_normal_vol={vol:.4f}")
print(
    f"call={call:.8f} put={put:.8f} "
    f"parity_error={call - put - discfactor * (forward - strike):.3e}"
)
print(f"delta={delta:.8f} finite_difference={delta_fd:.8f}")
print(f"implied_absolute={implied:.10f} error={implied - vol:.3e}")
```

The independent checks are put-call parity, a central delta difference with absolute
normal volatility held fixed, and a price-to-IV round trip. The normal IV solver's
default absolute-volatility bracket is `[1e-8, 1e4]`; use narrower market-specific bounds
when appropriate.

## Domain and failure boundaries

- `ttm` and `vol` must be positive in documented use. The normal pricer has no separate
  intrinsic branch; zero maturity or zero scale can divide by zero.
- Zero and negative forwards are supported because the volatility scale does not depend
  on the forward.
- Prices, strikes, forwards, and `vol` must share one price unit. Scaling only one of them
  changes the problem.
- The package returns absolute normal implied volatility in the same units.
- `IC`/`IP` do not add inverse settlement conversion. See the
  [inverse-option workflow](inverse_options.md).

See the [pricing guide](pricing_and_greeks.md), [implied-volatility guide](implied_volatility.md),
[API inventory](api.md), and
[Bachelier source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/bachelier.py).
