# Changelog

Entries start at 1.2.4. For earlier releases see the git log.

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
