---
myst:
  html_meta:
    description: >-
      Understand approximation, implied-volatility, intrinsic-boundary, and cold-versus-warm
      timing behavior in vanilla-option-pricers.
---

# Numerical accuracy and performance

Use this page to interpret numerical output and timing. Function success does not prove that an
input is well conditioned, and a warm timing does not describe first-call latency.

## Distribution functions

The package avoids SciPy so the numerical helpers can compile under Numba:

| Helper | Implementation contract | Consequence |
|---|---|---|
| `erfcc` / `ncdf` | Platform C-library `erfc`, called directly from a Numba ufunc; `ncdf` uses `0.5 * erfc(-x / sqrt(2))`. | Avoids cancellation in the lower normal tail and returns exactly `1` / `0.5` at the erfc/CDF origins. Accuracy follows the platform `libm`, normally near double precision. |
| `npdf` | Direct `exp(-x*x/2) / sqrt(2*pi)` evaluation. | Subject to ordinary floating-point underflow in extreme tails. |
| Internal `ncdf_inv` / `inv_erf` | Acklam piecewise rational normal quantile plus one lower-tail Halley refinement; `inv_erf` adds a central series and tail-stable mapping. | Near-double-precision delta-to-strike conversion across the representable probability range. |

The `erfcc`, `ncdf`, `ncdf_inv`, and `inv_erf` ufuncs do not enable Numba `fastmath`. `npdf` and
the Bachelier kernels retain their existing `fastmath=True` compilation. Do not assume bitwise
identity across platforms: the C-library special functions and generated machine code can differ.

## Executed public-API check

```python
from statistics import NormalDist

import numpy as np

from vanilla_option_pricers import (
    compute_bsm_strike_from_delta,
    compute_bsm_vanilla_price,
    infer_bsm_implied_vol,
    ncdf,
)

reference = NormalDist()
x_grid = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
cdf_error = max(
    abs(float(ncdf(x)) - reference.cdf(float(x))) for x in x_grid
)

forward = 100.0
ttm = 0.5
vol = 0.2
sdev = vol * np.sqrt(ttm)
delta_grid = np.array([0.001, 0.01, 0.1, 0.9, 0.99, 0.999])
approx_strikes = np.array(
    [compute_bsm_strike_from_delta(ttm, forward, float(d), vol) for d in delta_grid]
)
reference_strikes = np.array(
    [
        forward
        * np.exp(-sdev * (reference.inv_cdf(float(d)) - 0.5 * sdev))
        for d in delta_grid
    ]
)
strike_abs_error = np.max(np.abs(approx_strikes - reference_strikes))
strike_rel_error = np.max(
    np.abs((approx_strikes - reference_strikes) / reference_strikes)
)

intrinsic = compute_bsm_vanilla_price(
    105.0, 100.0, 0.0, 0.2, "C", 0.95
)
outside = infer_bsm_implied_vol(100.0, 1.0, 100.0, 1e6)
zero_iterations = infer_bsm_implied_vol(
    100.0, 1.0, 100.0, 10.0, max_iters=0
)

print(
    f"ncdf_max_abs={cdf_error:.3e} "
    f"strike_max_abs={strike_abs_error:.3e} "
    f"strike_max_rel={strike_rel_error:.3e}"
)
print(
    f"bsm_intrinsic={intrinsic:.6f} "
    f"unbracketed_is_nan={np.isnan(outside)}"
)
print(f"zero_iters={zero_iterations:.6f}")
```

Expected output in the verified environment:

```text
ncdf_max_abs=5.551e-17 strike_max_abs=2.842e-14 strike_max_rel=2.025e-16
bsm_intrinsic=5.000000 unbracketed_is_nan=True
zero_iters=2.505000
```

The CDF observation is a finite grid check, not a new global bound. A separate executed sweep of
229,117 probabilities from `1e-300` through the central interval and representable upper tail
found maximum quantile absolute error `2.843e-14` against `statistics.NormalDist.inv_cdf`, and
maximum probability round-trip error `2.220e-16`. Results can vary slightly with the platform
math library. `max_iters=0` returns the initial bracket midpoint (`2.505` for the BSM defaults);
it does not establish convergence.

## Intrinsic and degenerate boundaries

BSM `is_intrinsic` is true when `ttm <= 0`, `vol <= 0`, or `vol` is `nan`.

- BSM vanilla price returns raw `max(F-K, 0)` or `max(K-F, 0)` on that branch; the current branch
  does not apply `discfactor`.
- BSM delta returns its raw `0`, `1`, or `-1` limit; gamma, vega, and theta return zero. Digital
  price returns a raw `0`/`1` indicator.
- Bachelier functions do not share this guard. Zero and negative forwards are supported, but
  maturity and absolute volatility must produce a positive standard deviation; a zero scale can
  divide by zero.

These are current source behaviors, not settlement-policy recommendations. Handle expired
contracts and discounting deliberately outside the package when the raw intrinsic convention is
not the desired one.

## Implied-volatility solver limits

| Contract | BSM | Absolute Bachelier |
|---|---:|---:|
| Default bracket | `[0.01, 5.0]` | `[1e-8, 1e4]` |
| Default `tol` | `1e-8` absolute vol-step | `1e-8` absolute vol-step |
| Default `max_iters` | `100` | `100` |
| Unbracketed/non-positive target | `nan` by default | `nan` by default |

Both scalar solvers use safeguarded Newton steps with bisection fallback. Vanilla in-the-money
`C`/`P` targets are converted to the out-of-the-money parity counterpart; `IC`/`IP` targets are
inverted directly. With `is_bounds_to_nan=False`, an unbracketed target returns the violated bound
rather than proof of a root.

Low vega, very short maturity, deep moneyness, near-intrinsic prices, approximation error, and
rounded market quotes can dominate a small `tol`. Always reprice the returned volatility and
inspect the price residual. See the [IV workflow](implied_volatility.md) and
[Bachelier unit conversion](bachelier_convention.md).

## Cold and warm timing

First-call Numba compilation is separate from warm execution. A reproducible timing report must
record hardware, OS, Python, NumPy, Numba and package versions; function and signature; array
shape and dtype; random seed or fixed inputs; warm-up policy; repetitions; and reported statistic.
Measure cold latency in a fresh process, then warm the exact signature before repeated timing.

`examples/performance/bsm_speed.py` is a development diagnostic, not a CI correctness gate or a
universal benchmark. It prices a fixed 61-strike call chain with forward `100`, six-month
maturity, discount factor `0.98`, and an approximately 20% volatility smile. It prints three
sample prices, the first-call time including Numba compilation, the median warm time across five
repeats of 1,000 chains, and warm option prices per second. It also times the same inputs through
the compiled slice function and the `numpy.vectorize` convenience wrapper, checks that their
prices agree, reports their warm speed ratio, and records the platform and package versions. No
competitor or universal speed claim follows from that output. The module explicitly opts out of
pytest collection.

See the [array and Numba guide](array_shapes_and_numba.md), [API inventory](api.md),
[numerical helper source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/utils.py),
and [manual timing source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/examples/performance/bsm_speed.py).
