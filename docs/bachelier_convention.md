---
myst:
  html_meta:
    description: >-
      Convert absolute and relative Bachelier normal volatility correctly for
      vanilla-option-pricers.
---

# Bachelier normal-volatility convention

Use this page before passing a normal volatility to `compute_normal_price` or an
`infer_normal_*` function.

```{warning}
`vol` is **relative and dimensionless** in this package. It is not the absolute normal
volatility commonly accepted by Bachelier APIs.
```

For a positive forward $F$, the package defines

$$
\sigma_N = F\,vol, \qquad sdev = \sigma_N\sqrt{T} = F\,vol\sqrt{T},
$$

where `vol` is annualised relative normal volatility, $\sigma_N$ is annualised absolute normal
volatility in price units per square-root year, and `sdev` is the standard deviation over the
option's maturity.

## Convert an external quote

| Starting quote | Package input |
|---|---|
| Absolute normal volatility $\sigma_N$ | `vol = sigma_n / forward` |
| Maturity standard deviation `sdev` | `vol = sdev / (forward * sqrt(ttm))` |
| Package relative volatility `vol` | `sigma_n = forward * vol` |

These conversions require `forward > 0` and, for `sdev`, `ttm > 0`. The package does not infer
units or inspect vendor metadata. A vendor value of `5.0` in price units is `0.05` for a forward
of `100`, not `5.0` as a package input.

The analytic normal delta is the standard Bachelier derivative with **absolute** $\sigma_N$ held
fixed. When checking delta by shifting the forward, recompute `vol = sigma_n / shifted_forward`;
holding the relative `vol` fixed changes $\sigma_N$ and tests a different derivative.

## Executed conversion and reference checks

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
sigma_n = 5.0
vol = sigma_n / forward

call = compute_normal_price(forward, strike, ttm, vol, discfactor, "C")
put = compute_normal_price(forward, strike, ttm, vol, discfactor, "P")
delta = compute_normal_delta(ttm, forward, strike, vol, "C", discfactor)

h = 1e-4
price_up = compute_normal_price(
    forward + h, strike, ttm, sigma_n / (forward + h), discfactor, "C"
)
price_down = compute_normal_price(
    forward - h, strike, ttm, sigma_n / (forward - h), discfactor, "C"
)
delta_fd = (price_up - price_down) / (2.0 * h)

implied = infer_normal_implied_vol(
    forward, ttm, strike, call, discfactor, "C"
)

print(f"absolute_normal_vol={sigma_n:.4f} relative_vol={vol:.6f}")
print(
    f"call={call:.8f} put={put:.8f} "
    f"parity_error={call - put - discfactor * (forward - strike):.3e}"
)
print(f"delta={delta:.8f} finite_difference={delta_fd:.8f}")
print(
    f"implied_relative={implied:.10f} "
    f"implied_absolute={forward * implied:.10f} "
    f"error={implied - vol:.3e}"
)
```

Expected output:

```text
absolute_normal_vol=5.0000 relative_vol=0.050000
call=0.61771265 put=2.57771265 parity_error=0.000e+00
delta=0.28008772 finite_difference=0.28008773
implied_relative=0.0500000000 implied_absolute=5.0000000000 error=-6.939e-18
```

The independent checks are put-call parity, a central delta difference with $\sigma_N$ fixed,
and a price-to-IV round trip. The normal IV solver's default relative-volatility bracket
`[0.01, 10.0]` corresponds to the forward-dependent absolute bracket
`[0.01 * forward, 10.0 * forward]`.

## Domain and failure boundaries

- Documented use requires positive `forward`, `ttm`, and `vol`. Unlike the BSM functions, the
  normal pricer has no separate intrinsic branch; zero maturity or zero scale can divide by zero.
- `forward <= 0` is outside this relative conversion contract even though some absolute-normal
  model formulations permit non-positive forwards.
- Prices, strikes, and $\sigma_N$ must share one price unit. Scaling only one of them changes the
  problem.
- The package returns relative normal implied volatility. Multiply by the same forward used in
  the inversion to recover $\sigma_N$.
- `IC`/`IP` do not add inverse settlement conversion. See the
  [inverse-option workflow](inverse_options.md).

See the [pricing guide](pricing_and_greeks.md), [implied-volatility guide](implied_volatility.md),
[API inventory](api.md), and
[Bachelier source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/bachelier.py).
