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

## Distribution approximations

The package avoids SciPy so the numerical helpers can compile under Numba:

| Helper | Implementation contract | Consequence |
|---|---|---|
| `ncdf` | Numerical Recipes complementary-error-function rational approximation; the source states a conservative maximum absolute error of `1.2e-7`. | The bound propagates into prices, Greeks, and IV residuals. |
| `npdf` | Direct `exp(-x*x/2) / sqrt(2*pi)` evaluation. | Subject to ordinary floating-point underflow in extreme tails. |
| Internal `ncdf_inv` | Winitzki inverse-erf approximation; the source documents approximately `1.3e-4` relative error. | Exported delta-to-strike helpers are coarse in tail delta grids. |

The numerical helpers and Bachelier kernels use Numba `fastmath=True`; do not assume strict IEEE
operation ordering or bitwise identity across platforms. The BSM kernels use their current Numba
decorators without a cross-platform bitwise-reproducibility promise.

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
ncdf_max_abs=1.500e-08 strike_max_abs=1.236e-01 strike_max_rel=7.912e-04
bsm_intrinsic=5.000000 unbracketed_is_nan=True
zero_iters=2.505000
```

The CDF observation is a finite grid check, not a new global bound. The strike error shows the
public effect of the coarser inverse-CDF approximation. `max_iters=0` returns the initial bracket
midpoint (`2.505` for the BSM defaults); it does not establish convergence.

## Intrinsic and degenerate boundaries

BSM `is_intrinsic` is true when `ttm <= 0`, `vol <= 0`, or `vol` is `nan`.

- BSM vanilla price returns raw `max(F-K, 0)` or `max(K-F, 0)` on that branch; the current branch
  does not apply `discfactor`.
- BSM delta returns its raw `0`, `1`, or `-1` limit; gamma, vega, and theta return zero. Digital
  price returns a raw `0`/`1` indicator.
- Bachelier functions do not share this guard. Documented normal-model use requires positive
  forward, maturity, and relative volatility; a zero scale can divide by zero.

These are current source behaviors, not settlement-policy recommendations. Handle expired
contracts and discounting deliberately outside the package when the raw intrinsic convention is
not the desired one.

## Implied-volatility solver limits

| Contract | BSM | Relative Bachelier |
|---|---:|---:|
| Default bracket | `[0.01, 5.0]` | `[0.01, 10.0]` |
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
universal benchmark. It fixes seed `3`, creates 10,000-element arrays, warms the slice, vector,
and grid paths once, and prints one average from 20 calls. It does not record the full environment
or a distribution across repetitions, and the three helpers have different shape contracts. No
competitor or universal speed claim follows from that output.

See the [array and Numba guide](array_shapes_and_numba.md), [API inventory](api.md),
[numerical helper source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/utils.py),
and [manual timing source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/examples/performance/bsm_speed.py).
