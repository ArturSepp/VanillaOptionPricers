---
myst:
  html_meta:
    description: >-
      Price Black-Scholes-Merton and Bachelier vanilla options and compute forward Greeks with
      vanilla-option-pricers.
---

# Pricing and Greeks

Use this page when you already have a forward, discount factor, maturity, strike, volatility, and
option type. The package prices the supplied forward payoff; it does not build spots, forwards,
rates, dividends, discount curves, calendars, or settlement conventions.

## Inputs and conventions

| Input | Type and units | Contract |
|---|---|---|
| `forward` | `float`, price units | Caller-supplied forward (F). BSM requires a positive forward; Bachelier also supports zero and negative forwards. |
| `strike` | `float`, same units as `forward` | Strike (K). BSM requires a positive strike. |
| `ttm` | `float`, years | Time to maturity (T). Documented examples use positive values. |
| `vol` for BSM | `float`, annualised decimal | Lognormal volatility; `0.20` means 20%. |
| `vol` for Bachelier | positive `float`, annualised price/rate units | Absolute normal volatility; the period standard deviation is (vol\sqrt{T}). |
| `discfactor` | `float`, default `1.0` | Discount factor (D), conventionally (\exp(-rT)). |
| `optiontype` | `str` | `"C"` or `"P"` for the documented vanilla workflows. `"IC"` and `"IP"` select the corresponding internal payoff branch, but their quote and numeraire conversion is caller-owned and documented separately in V4b. |
| `discount_rate` | `float`, default `0.0` | Continuously compounded (r), used only by BSM theta's rate term. |

Black-Scholes-Merton uses (D(FN(d_1)-KN(d_2))) for a call and the corresponding put expression.
Bachelier uses annualised absolute normal volatility. Both models satisfy
(C-P=D(F-K)) for consistent inputs.

## First pricing and Greek result

This example constructs a zero-dividend forward and discount factor outside the package, then
uses only package-root functions.

```python
import numpy as np

from vanilla_option_pricers import (
    compute_bsm_vanilla_delta,
    compute_bsm_vanilla_gamma,
    compute_bsm_vanilla_price,
    compute_bsm_vanilla_slice_prices,
    compute_bsm_vanilla_theta,
    compute_bsm_vanilla_vega,
    compute_normal_price,
)

spot = 100.0
rate = 0.04
ttm = 0.50
forward = spot * np.exp(rate * ttm)
discfactor = np.exp(-rate * ttm)
strike = 105.0
vol = 0.20

call = compute_bsm_vanilla_price(forward, strike, ttm, vol, "C", discfactor)
put = compute_bsm_vanilla_price(forward, strike, ttm, vol, "P", discfactor)
delta_f = compute_bsm_vanilla_delta(ttm, forward, strike, vol, "C", discfactor)
vega = compute_bsm_vanilla_vega(ttm, forward, strike, vol, discfactor)
gamma_f = compute_bsm_vanilla_gamma(ttm, forward, strike, vol)
theta = compute_bsm_vanilla_theta(
    ttm, forward, strike, vol, "C", discfactor, rate
)
normal_call = compute_normal_price(forward, strike, ttm, 20.0, discfactor, "C")

strikes = np.array([95.0, 100.0, 105.0])
vols = np.array([0.18, 0.20, 0.22])
optiontypes = np.array(["P", "C", "C"])
slice_prices = compute_bsm_vanilla_slice_prices(
    ttm, forward, strikes, vols, optiontypes, discfactor
)

print(f"forward={forward:.4f} discount={discfactor:.6f}")
print(f"call={call:.4f} put={put:.4f} delta_F={delta_f:.6f}")
print(f"vega={vega:.4f} gamma_F={gamma_f:.8f} theta={theta:.4f}")
print(f"normal_call={normal_call:.4f}")
print("slice=" + np.array2string(slice_prices, precision=4))
```

Expected output:

```text
forward=102.0201 discount=0.980199
call=4.3770 put=7.2979 delta_F=0.438295
vega=27.9616 gamma_F=0.02740790 theta=-7.2058
normal_call=4.1921
slice=[2.2069 6.6271 4.9371]
```

`delta_F` is the derivative of the discounted price with respect to the supplied forward. If the
caller uses (F=S\exp((r-q)T)), the corresponding spot derivative is
`delta_F * exp((r - q) * ttm)`. `compute_bsm_vanilla_gamma` is an undiscounted forward gamma, so
the second derivative of the discounted price is `discfactor * gamma_F`.

The scalar BSM vega accepts `discfactor`; its default `1.0` is the undiscounted vega. BSM
`compute_bsm_slice_vegas` and `compute_bsm_vegas_ttms`, and normal
`compute_normal_slice_vegas` and `compute_normal_vegas_ttms`, are undiscounted. Multiply their
results by the applicable discount factor when the required derivative is of the discounted
price. BSM theta is the sign-flipped maturity derivative (calendar decay) with spot held fixed;
the example uses the consistent zero-dividend forward and supplies `rate` explicitly.

## Choose the execution shape

| Shape | Functions | Input contract |
|---|---|---|
| Scalar | `compute_bsm_vanilla_price`, scalar BSM Greeks and digitals; `compute_normal_price`, `compute_normal_delta` | Scalar values; Numba-compiled. |
| Convenience vector | Five BSM `*_vector` names | `numpy.vectorize` wrappers around scalar functions. They are convenience dispatchers, not compiled array kernels or a performance guarantee. |
| Expiry slice | BSM and normal `*_slice_*` names | One expiry and forward; aligned strike, volatility, and option-code arrays. Equal logical lengths are caller-owned. |
| Forward grid | `compute_bsm_forward_grid_prices`, `compute_bsm_vanilla_grid_deltas` | One strike/vol/type evaluated across a forward array. |
| Expiry chain | BSM and normal `*_ttms` names | `ttms` and `forwards` aligned by expiry, with one `numba.typed.List` array per expiry. |

Arbitrary broadcasting is not part of the package contract. Prefer the named scalar, aligned
slice, grid, or chain helper that matches the data layout.

## Failure behavior and independent checks

- Unsupported option codes raise `NotImplementedError` in price/theta/digital functions; normal
  delta returns `nan` for an unsupported code. Validate codes before a bulk call.
- BSM price and Greeks switch to their intrinsic/zero-diffusion branches when `ttm <= 0`,
  `vol <= 0`, or BSM `vol` is `nan`. The normal functions support non-positive forwards but
  expect positive maturity and absolute volatility and do not provide the same intrinsic guard.
- Slice and chain functions assume aligned inputs and do not perform a separate shape-validation
  pass. A short input can truncate a `zip`; an incompatible dtype can fail during Numba typing.
- The first call for a new Numba signature includes compilation time.

For the example above, the executed put-call-parity residual was exactly `0.0`. Central finite
differences produced delta `0.43829106`, discounted gamma `0.02686534`, vega `27.96150047`, and
theta `-7.20580075`; the analytic values were `0.43829531`, `0.02686519`, `27.96157666`, and
`-7.20583331`. The small differences are consistent with finite-difference truncation and
floating-point rounding, rather than evidence from successful execution alone.

See the [complete API inventory](api.md),
[BSM source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/black_scholes.py),
and [Bachelier source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/bachelier.py).
