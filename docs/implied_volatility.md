---
myst:
  html_meta:
    description: >-
      Recover scalar, slice, and option-chain implied volatility with vanilla-option-pricers.
---

# Implied volatility

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/VanillaOptionPricers/commit/ae0e184c55b30824bfc957ffd7eb7e3d0a9749d5)*

Implemented in [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers).
Software citation: [CITATION.cff](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff).

Use implied-volatility functions when an observed or model price, forward, discount factor,
maturity, strike, and option type are already expressed in one consistent convention. BSM returns
annualised lognormal volatility. Bachelier returns annualised absolute normal volatility in the
same units as the forward and strike.

## Scalar solver contract

The Black solver uses Peter Jaeckel's transformed rational initial guess followed by
Householder refinement, adapted from [Let's Be Rational](http://www.jaeckel.org/LetsBeRational.pdf),
reference revision 1520 (2024). It normally needs two corrections; exactly ATM prices use an
inverse-error-function formula. The Bachelier solver retains safeguarded Newton iteration with
bisection fallback. `tol` is an absolute volatility-step tolerance, not a price or relative-error
tolerance. `max_iters` limits refinements; an exhausted limit returns the last iterate and does
not certify convergence. The Black initial guess is independent of the caller's bounds, which
are applied to the resulting volatility. These numerical changes are unreleased.

| Argument | BSM default | Normal default | Meaning |
|---|---:|---:|---|
| `tol` | `1e-8` | `1e-8` | Absolute volatility-step convergence tolerance. |
| `vol_lower` | `0.01` | `1e-8` | Lower bracket endpoint. |
| `vol_upper` | `5.0` | `1e4` | Upper bracket endpoint. |
| `max_iters` | `100` | `100` | Maximum iterative corrections. |
| `is_bounds_to_nan` | `True` | `True` | Return `nan` for an unbracketed/non-positive target; when `False`, return the violated bracket bound. |

In-the-money prices are transformed to their out-of-the-money counterparts using
put-call parity. Both solvers then evaluate the OTM formula directly. `"IC"` and `"IP"` continue
to share the respective call and put payoff branches; callers still own external inverse-quote
normalisation. No solver can recover time value already lost when an ITM input price was rounded.

## Scalar, slice, and chain example

The example uses `numba.typed.List` strings. Fixed-width NumPy Unicode option-code arrays are
also supported by the compiled price and IV loops, including the `"IC"` and `"IP"` codes.

```python
import numpy as np
from numba.typed import List

from vanilla_option_pricers import (
    compute_bsm_vanilla_price,
    compute_bsm_vanilla_slice_prices,
    compute_normal_price,
    infer_bsm_implied_vol,
    infer_bsm_ivols_from_model_chain_prices,
    infer_bsm_ivols_from_slice_prices,
    infer_normal_implied_vol,
)

forward, discfactor, ttm, strike, vol = 101.25, 0.99, 0.25, 105.0, 0.20
normal_vol = 20.0
bsm_price = compute_bsm_vanilla_price(
    forward, strike, ttm, vol, "C", discfactor
)
bsm_iv = infer_bsm_implied_vol(
    forward, ttm, strike, bsm_price, discfactor, "C"
)
normal_price = compute_normal_price(
    forward, strike, ttm, normal_vol, discfactor, "C"
)
normal_iv = infer_normal_implied_vol(
    forward, ttm, strike, normal_price, discfactor, "C"
)

strikes = np.array([95.0, 101.25, 105.0])
vols = np.array([0.18, 0.20, 0.22])
price_types = np.array(["P", "C", "C"])
solve_types = List(["P", "C", "C"])
prices = compute_bsm_vanilla_slice_prices(
    ttm, forward, strikes, vols, price_types, discfactor
)
slice_iv = infer_bsm_ivols_from_slice_prices(
    ttm, forward, discfactor, strikes, solve_types, prices
)

ttms = np.array([0.25, 0.75])
forwards = np.array([101.25, 103.0])
discfactors = np.array([0.99, 0.97])
target_vols = [np.array([0.18, 0.22]), np.array([0.19, 0.24])]
strike_arrays = [np.array([95.0, 105.0]), np.array([90.0, 110.0])]
type_arrays = [np.array(["P", "C"]), np.array(["P", "C"])]

strikes_ttms = List()
optiontypes_ttms = List()
prices_ttms = List()
for T, F, D, K, sigma, optiontype in zip(
    ttms, forwards, discfactors, strike_arrays, target_vols, type_arrays
):
    strikes_ttms.append(K)
    optiontypes_ttms.append(List(optiontype.tolist()))
    prices_ttms.append(
        compute_bsm_vanilla_slice_prices(T, F, K, sigma, optiontype, D)
    )

chain_iv = infer_bsm_ivols_from_model_chain_prices(
    ttms,
    forwards,
    discfactors,
    strikes_ttms,
    optiontypes_ttms,
    prices_ttms,
)

print(f"bsm_price={bsm_price:.6f} bsm_iv={bsm_iv:.8f}")
print(f"normal_price={normal_price:.6f} normal_iv={normal_iv:.8f}")
print("slice_iv=" + np.array2string(slice_iv, precision=8))
print("chain_iv=" + str([np.round(np.asarray(x), 8).tolist() for x in chain_iv]))
```

Expected output:

```text
bsm_price=2.481053 bsm_iv=0.20000000
normal_price=2.367771 normal_iv=20.00000000
slice_iv=[0.18 0.2  0.22]
chain_iv=[[0.18, 0.22], [0.19, 0.24]]
```

The executed maximum absolute round-trip error was `2.53e-15` across the scalar, slice, and chain
values. This is a generated, identifiable regime; low-vega inputs will not generally recover to
that many digits.

## Choose the inversion workflow

| Workflow | BSM | Bachelier | Container/default behavior |
|---|---|---|---|
| Scalar | `infer_bsm_implied_vol` | `infer_normal_implied_vol` | Exposes tolerance, bracket, iteration limit, and bound-to-`nan` policy. |
| Slice, configurable | `infer_bsm_ivols_from_model_slice_prices` | `infer_normal_ivols_from_model_slice_prices` | Aligned one-expiry inputs. BSM exposes bracket/iteration/bound controls; normal uses scalar defaults. |
| Slice, discfactor-first | `infer_bsm_ivols_from_slice_prices` | `infer_normal_ivols_from_slice_prices` | Aligned one-expiry convenience ordering; uses scalar defaults. |
| Chain | `infer_bsm_ivols_from_model_chain_prices` | `infer_normal_ivols_from_chain_prices` | One forward/discount factor per expiry and one typed container per expiry; uses scalar defaults. |

`ttms`, `forwards`, and `discfactors` must have matching expiry order. Each per-expiry strike,
option-code, and price container must be aligned. The helpers allocate one output array per expiry
and do not perform a separate schema-validation pass.

## Failure behavior and conditioning

- A `nan`, non-positive, or unbracketed scalar target returns `nan` when
  `is_bounds_to_nan=True`. With it disabled, the scalar solver returns `vol_lower` or `vol_upper`.
  An executed BSM target price of `1e6` returned `nan`; with bounds enabled it returned the upper
  bound `5.0`.
- `infer_bsm_ivols_from_model_slice_prices` maps `nan` and near-zero prices to `nan` (or the lower
  bound). The normal slice helpers delegate each element to the scalar normal solver.
- Deep in/out-of-the-money, very short-maturity, or nearly intrinsic prices have low vega. A small
  price error can then imply a large volatility error, and a tight `tol` does not create missing
  information.
- A too-narrow bracket is an input-policy failure, not evidence that the market has no implied
  volatility. Expand it deliberately and retain economically meaningful bounds.
- Invalid option codes or incompatible array/string container types fail in the underlying pricer
  or during Numba compilation. Validate them before bulk inversion.

See the [complete API inventory](api.md),
[BSM source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/black_scholes.py),
and [Bachelier source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/bachelier.py).
