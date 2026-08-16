---
myst:
  html_meta:
    description: >-
      Choose scalar, vector, slice, grid, and chain APIs and understand Numba behavior in
      vanilla-option-pricers.
---

# Array shapes and Numba behavior

Use the helper whose data layout matches the task. The package has several distinct execution
paths; it does not promise arbitrary broadcasting across all functions.

## Shape contract

| Path | Inputs | Output | Execution behavior |
|---|---|---|---|
| Scalar | Python/NumPy scalars plus one Python option-code string | Scalar | Numba compiles a signature on first use. |
| Five BSM `*_vector` wrappers | NumPy-broadcastable scalar arguments | Broadcast array | `numpy.vectorize` calls the scalar dispatcher element by element. |
| Slice | One scalar maturity/forward plus aligned 1-D strike, vol, type, or price containers | One array shaped like the allocation input, normally strikes | Numba-compiled loop or array expression. |
| Forward grid | One option contract plus a 1-D forward array | Array shaped like `forwards` | Numba-compiled loop. |
| Chain | Aligned `ttms`, `forwards`, and sometimes `discfactors`; one typed slice per expiry | `numba.typed.List` of arrays | Numba-compiled outer and inner loops. |

Slice and chain helpers use `zip` and do not run a schema-validation pass. Unequal lengths can
truncate iteration and leave preallocated output entries unchanged. Validate every aligned
dimension before calling.

Many compiled helpers allocate with `np.zeros_like(strikes)` or `np.zeros_like(forwards)`. Use
floating-point numeric arrays: integer allocation inputs can cause computed prices or Greeks to
be cast back to integers. One-dimensional `float64` arrays are the conservative default.

## Executed shape example

```python
import numpy as np
from numba.typed import List

from vanilla_option_pricers import (
    compute_bsm_forward_grid_prices,
    compute_bsm_vanilla_deltas_ttms,
    compute_bsm_vanilla_price_vector,
    compute_bsm_vanilla_slice_prices,
)

forward_matrix = np.array([[95.0], [105.0]])
strike_matrix = np.array([[90.0, 100.0, 110.0]])
vector_prices = compute_bsm_vanilla_price_vector(
    forward_matrix, strike_matrix, 0.5, 0.2, "C", 1.0
)

strikes = np.array([90.0, 100.0, 110.0])
vols = np.array([0.2, 0.2, 0.2])
optiontypes = np.array(["P", "C", "C"])
slice_prices = compute_bsm_vanilla_slice_prices(
    0.5, 100.0, strikes, vols, optiontypes
)
grid_prices = compute_bsm_forward_grid_prices(
    0.5, np.array([95.0, 100.0, 105.0]), 100.0, 0.2, "C"
)

ttms = np.array([0.25, 0.50])
forwards = np.array([100.0, 101.0])
strikes_ttms = List()
vols_ttms = List()
optiontypes_ttms = List()
for strike_slice, vol_slice, type_slice in (
    (np.array([95.0, 105.0]), np.array([0.2, 0.2]), np.array(["P", "C"])),
    (
        np.array([90.0, 100.0, 110.0]),
        np.array([0.2, 0.2, 0.2]),
        np.array(["P", "C", "C"]),
    ),
):
    strikes_ttms.append(strike_slice)
    vols_ttms.append(vol_slice)
    optiontypes_ttms.append(type_slice)

chain_deltas = compute_bsm_vanilla_deltas_ttms(
    ttms, forwards, strikes_ttms, vols_ttms, optiontypes_ttms
)

print(
    f"vector_shape={vector_prices.shape} "
    f"slice_shape={slice_prices.shape} grid_shape={grid_prices.shape}"
)
print("chain_shapes=" + str([np.asarray(x).shape for x in chain_deltas]))
print(
    f"dtypes={vector_prices.dtype},{slice_prices.dtype},{grid_prices.dtype}"
)
```

Expected output:

```text
vector_shape=(2, 3) slice_shape=(3,) grid_shape=(3,)
chain_shapes=[(2,), (3,)]
dtypes=float64,float64,float64
```

Only the five BSM `*_vector` wrappers use ordinary NumPy broadcasting in this example. A
`numpy.vectorize` wrapper is a convenience loop, not a compiled array kernel and not evidence of
a speed advantage. Slice, grid, and chain helpers have the explicit layouts in the table.

## Option-code containers

- Scalar kernels take a Python string such as `"C"` or `"P"`.
- Pricing slice helpers accept one option code per strike; fixed-width NumPy Unicode arrays work
  in the currently tested price paths.
- Current BSM and normal IV slice/chain compilation can fail when a fixed-width NumPy Unicode
  array reaches the parity-conditioning assignment. Use dynamic strings, including
  `numba.typed.List(["P", "C"])`; for a chain use an outer typed list containing one inner typed
  string list per expiry. The [IV example](implied_volatility.md) shows that layout.
- Validate option codes before bulk calls. Error behavior is not uniform: price/theta/digital
  functions raise for unsupported codes, normal delta returns `nan`, and current BSM non-intrinsic
  delta maps codes other than `C`/`P` to zero.

## First-call compilation

The scalar, slice, grid, and chain kernels are Numba dispatchers. The first call for each new
combination of argument types compiles native code; later calls with the same signature are warm.
Changing a float to an integer, changing array dtype, or changing a container type can create a
new signature or a typing failure. The current decorators do not provide a persistent disk-cache
contract, so a new Python process should be treated as cold.

Do not compare a cold Numba call with a warm call or with a `numpy.vectorize` wrapper and label the
difference as algorithmic performance. See [numerical accuracy and timing](numerical_accuracy_and_performance.md)
for the measurement contract.

See the [pricing guide](pricing_and_greeks.md), [API inventory](api.md),
[BSM source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/black_scholes.py),
and [Bachelier source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/bachelier.py).
