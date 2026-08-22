## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results. For numerical changes, name the independent
reference-value, parity, or finite-difference cross-check.

## Checklist

- [ ] Tests cover the changed behavior or defect.
- [ ] Forward, discount-factor, expiry, payoff, volatility, and array-shape conventions remain explicit.
- [ ] Numerical results were checked independently; expected values were not changed to fit output.
- [ ] The public API remains function-based and broadcasting claims match supported paths.
- [ ] No runtime dependency beyond NumPy and Numba was added.
- [ ] No proprietary data, local paths, generated output, or development environments are included.
- [ ] `uv run --no-sync pytest` and the relevant lint/docs/wheel checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] Public-signature, default, dependency-floor, or runtime-dependency changes are called out explicitly.
