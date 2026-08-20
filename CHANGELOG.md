# Changelog

Entries start at 1.2.4. For earlier releases see the git log.

## [Unreleased]

## [2.1.0] - 2026-08-20

### Added
- Added a dedicated Bachelier workflow example covering absolute-normal quote conversion,
  scalar pricing and inversion, delta-to-strike conversion, aligned slice Greeks, and negative
  rate forwards.

### Changed
- Reworked `examples/performance/bsm_speed.py` into a plain 61-strike call-chain example with
  sample prices, human-readable cold/warm timing, throughput, and an equal-input comparison of
  the compiled Numba batch function with the `numpy.vectorize` wrapper. It records environment
  metadata and explicitly opts out of pytest collection.
- `erfcc` and `ncdf` now call the platform `erfc` implementation directly from Numba and evaluate
  the normal CDF in a lower-tail-stable form, removing the former `1.2e-7` approximation ceiling
  and the discontinuity at zero.
- `inv_erf` and `ncdf_inv` now use an Acklam piecewise rational quantile with a lower-tail Halley
  refinement and stable central/tail mappings. Independent checks cover probabilities to
  `1e-300` with near-double-precision agreement, improving `compute_bsm_strike_from_delta` and
  `compute_normal_delta_to_strike` without adding a dependency or changing their signatures.

## [2.0.0] - 2026-08-18

### Changed
- **Breaking:** all public Bachelier functions now consume and return annualised absolute normal
  volatility in forward/strike units. Version 1.x used dimensionless relative normal volatility
  with `absolute_vol = forward * vol`.
- `compute_normal_price`, normal delta/strike and vega helpers, and every normal implied-volatility
  inverter now use `sdev = vol * sqrt(ttm)` consistently. Zero and negative forwards are valid.
- `infer_normal_implied_vol` defaults now bracket absolute volatility on `[1e-8, 1e4]`.

### Fixed
- Removed the forward multiplier from absolute-normal vegas and safeguarded-Newton derivatives,
  aligning analytic Greeks and implied-volatility inversion with the standard Bachelier formula.
- Black and normal chain IV solvers now accept the fixed-width NumPy option-code arrays used by
  compiled SVM option chains without a Numba string-lowering failure.

## [1.3.1] - 2026-08-16

### Added
- Added a thin, output-free Colab entry point that installs the latest public PyPI release and is
  mechanically checked against the authoritative offline quickstart.
- Added one authoritative offline pricing and implied-volatility quickstart under root
  `examples/`, included directly in the hosted documentation and exercised from a clean wheel on
  Linux, Windows, and macOS CI.

### Changed
- Aligned the package metadata, README, citation title, and software BibTeX entry on one scope:
  Numba-vectorised Black-Scholes-Merton and Bachelier prices, Greeks, and implied-volatility
  fits over NumPy arrays for quantitative research pipelines.
- Clarified that the public pricing API consumes caller-supplied forwards and discount factors,
  and bounded the `IC`/`IP`, array execution, and performance descriptions to behavior supported
  by the current API.

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
