# Installation and a first result

## Install

Install the released package from PyPI:

```console
python -m pip install vanilla-option-pricers
```

For a source checkout, install the package and the build-only documentation tools separately:

```console
python -m pip install -e ".[docs]"
```

The package's runtime dependencies remain NumPy and Numba. Sphinx, MyST Parser, and Furo belong
only to the optional `docs` extra.

## Run the authoritative offline example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/VanillaOptionPricers/blob/main/notebooks/offline_quickstart_colab.ipynb)

The repository's
[pricing and IV script](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/examples/getting_started/pricing_and_iv.py)
is the single source for first success. It uses only the installed package, NumPy, Numba, and
deterministic generated inputs; it requires no network, credentials, pandas, or SciPy and creates
no files.

From a source checkout with the package installed, run:

```console
python examples/getting_started/pricing_and_iv.py
```

The wheel intentionally excludes repository examples. After installing from PyPI, download or
copy the linked script and run it from any directory.

For a zero-local-setup trial, the Colab notebook installs the latest released distribution from
the official PyPI index, prints its distribution version and import path, and runs the exact same
workflow below. The committed notebook contains no execution output.

```{literalinclude} ../examples/getting_started/pricing_and_iv.py
:language: python
:linenos:
```

The script prices one aligned BSM slice, recovers all four input volatilities, and checks the
same-strike call/put pair against forward put-call parity. It also prices and inverts one
Bachelier call with the package's relative normal-volatility convention:
`absolute_normal_vol = forward * relative_vol`.

Successful output reports package version and import path; input/output shapes; option codes and
prices; maximum IV and parity errors; one cold first-call duration; the mean of ten warm repeats;
and the Bachelier conversion. Timing values are machine-dependent and are not a benchmark. The
verified numerical errors are required to remain below the explicit thresholds in the script.

Start adaptation with the final `change_first` line: `forward`, `discfactor`, `ttm`, `strikes`,
`option_types`, and `model_convention`. Preserve the stated units and alignment contracts when
substituting market inputs.

## Next steps and boundaries

Use the package-root functions for stable user examples. Array helpers have distinct scalar,
aligned-array, grid, and per-expiry contracts; arbitrary broadcasting is not promised. The first
call to a Numba-compiled signature includes compilation time.

Continue with [pricing and Greeks](pricing_and_greeks.md),
[implied volatility](implied_volatility.md),
[Bachelier units](bachelier_convention.md), and
[array/Numba behavior](array_shapes_and_numba.md). Use the
[issue tracker](https://github.com/ArturSepp/VanillaOptionPricers/issues) for support.
