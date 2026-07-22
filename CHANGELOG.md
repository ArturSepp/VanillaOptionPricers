# Changelog

Entries start at 1.2.4. For earlier releases see the git log.

## [1.3.0] - 2026-07-22

### Changed
- `infer_bsm_implied_vol` and `infer_normal_implied_vol` now solve by safeguarded Newton
  (Newton-Raphson with bisection fallback) using the analytic vega, converging in roughly
  5-8 pricer evaluations instead of the previous fixed bisection sweep. Implied vols are
  unchanged where the volatility is identifiable; terminal precision is bounded by `ncdf`
  (~1e-7), which dominates only in the degenerate near-zero-vega regime (deep in/out of the
  money), where vol is not recoverable from a price.
- `infer_bsm_implied_vol`: `tol` now measures convergence on the implied vol (default
  `1e-8`) rather than on the price residual (was `1e-16`, unreachable); default `max_iters`
  is `100` (was `200`).
- `infer_normal_implied_vol`: signature aligned with `infer_bsm_implied_vol` -- adds
  `vol_lower`, `vol_upper`, `max_iters`; `is_bounds_to_nan` now defaults to `True` (was
  `False`); `tol` now measures convergence on the vol (default `1e-8`, was `1e-12` on the
  price residual). A price outside the achievable range returns `nan` instead of silently
  returning a bracket bound.
- Both inverters invert the out-of-the-money counterpart of a vanilla in-the-money option
  via put-call parity `C - P = discfactor*(F - K)` for conditioning; the returned implied
  vol is unchanged.
- `compute_bsm_vanilla_vega` (and `compute_bsm_vanilla_vega_vector`) take an optional
  `discfactor` (default `1.0`) and return `discfactor * F * n(d1) * sqrt(ttm)`. The default
  reproduces the prior undiscounted output exactly.

### Added
- `compute_bsm_digital_price` and `compute_bsm_digital_delta` are now exported from the
  package root.
- `vanilla_option_pricers/tests/test_bachelier.py`: numpy-only put-call parity,
  finite-difference greek, implied-vol round-trip, and delta-to-strike tests for the
  Bachelier model.
- NumPy-style docstrings on every function and a module header on `black_scholes`,
  `bachelier` and `utils`, including the `ncdf`/`ncdf_inv` accuracy contract.

### Removed
- Internal helpers `compute_bsm_vanilla_slice_vegas` and `compute_bsm_vanilla_vegas_ttms`,
  byte-for-byte duplicates of the exported `compute_bsm_slice_vegas` and
  `compute_bsm_vegas_ttms`. Neither was exported.

### Fixed
- `compute_normal_delta_to_strike` called `ncdf_inv.ppf(...)`, but `ncdf_inv` is a numba
  function with no `.ppf` attribute, so the function raised `AttributeError` on every call.
  It now calls `ncdf_inv` directly.
- `inv_erf` (and therefore `ncdf_inv`, `compute_bsm_strike_from_delta` and
  `compute_normal_delta_to_strike`) was both sign-wrong -- it dropped the `sign(x)` factor,
  so it returned `|erf^{-1}(x)|` and mapped a probability `p` and `1-p` to the same value --
  and magnitude-wrong -- it squared `2/(pi*a) + ln(1-x^2)` instead of
  `2/(pi*a) + ln(1-x^2)/2`. Strikes from deltas away from 0.5 were wrong in sign and size;
  `ncdf_inv` now matches the true quantile to the Winitzki bound (~1.3e-4).

## [1.2.4] - 2026-07-22

### Fixed
- `compute_bsm_vanilla_theta` (and `compute_bsm_vanilla_theta_vector`): the volatility-decay
  term was 4x too large -- `vol/(0.5*sqrt(ttm))` instead of `vol/(2*sqrt(ttm))` -- and omitted
  the leading `discfactor`, so theta was wrong in every regime for both calls and puts.
  Contributed by @gaoflow (#1).

### Added
- Scalar greeks `compute_bsm_vanilla_vega`, `compute_bsm_vanilla_gamma` and
  `compute_bsm_vanilla_theta` are now exported from the package root, matching the already
  exported `compute_bsm_vanilla_delta` and the `_vector` wrappers.
- `vanilla_option_pricers/tests/test_black_scholes.py`: put-call parity and finite-difference
  checks of delta, gamma, vega and theta against the pricer. Contributed by @gaoflow (#1).
