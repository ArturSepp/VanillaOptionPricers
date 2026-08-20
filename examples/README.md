# Examples

Examples import the installed public `vanilla_option_pricers` API. From the repository root,
install the project before running them:

```bash
pip install -e ".[dev]"
```

## Getting started

Deterministic, offline-first examples live in `getting_started/`. They require no credentials,
downloads, or optional data dependencies:

- `pricing_and_iv.py` is the shortest BSM/Bachelier first-success workflow.
- `normal_vol_workflow.py` shows absolute-normal units, price-to-IV recovery, forward-delta
  strike conversion, slice prices, deltas, vegas, and a negative-forward rate example.

Run them from the repository root:

```bash
python examples/getting_started/pricing_and_iv.py
python examples/getting_started/normal_vol_workflow.py
```

## Performance diagnostics

Scripts in `performance/` are opt-in timing diagnostics for maintainers. `bsm_speed.py` prices a
familiar 61-strike call chain, prints three sample prices, and explains first-run versus warm
speed in milliseconds, microseconds, and option prices per second. It then prices the identical
chain with the compiled Numba batch function and the `numpy.vectorize` convenience wrapper,
verifies equal prices, and reports their warm speed ratio. Results are machine-dependent and are
not correctness or CI gates. Run it with:

```bash
python examples/performance/bsm_speed.py
```

The root `examples/` directory is repository-only and is not included in the wheel.
