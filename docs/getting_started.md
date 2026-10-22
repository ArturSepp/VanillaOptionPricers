---
myst:
  html_meta:
    description: >-
      Install vanilla-option-pricers from PyPI and run the offline first example: a
      Black-Scholes-Merton slice priced and inverted to implied volatility, a put-call parity
      check, and a Bachelier call in normal volatility.
---

# Installation and a first result

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/VanillaOptionPricers/commit/ebf17a911688c0820c5082d66df3cd9131d554af)*

Implemented in [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers).
Software citation: [CITATION.cff](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff).

Price a fixed four-option slice and recover its implied volatilities on a CPU.

## Release boundary and reproducible first use

The verified release floor for this workflow is **vanilla-option-pricers 2.2.1** on Python
3.12. This is a tested support floor, not a claim about the first historical introduction.
The package remains Beta; retain the explicit model and shape conventions when upgrading.
[Stable documentation](https://vanillaoptionpricers.readthedocs.io/en/stable/) describes the
release channel; [latest](https://vanillaoptionpricers.readthedocs.io/en/latest/) follows main
and can describe unreleased changes. The rendered site's build version identifies its source;
a stable alias is not proof of the newest PyPI upload.

For a reproducible trial, install the core distribution without extras and save the existing
script from the verified source revision. Run these commands in a working directory and Python
environment outside synchronized folders:

```powershell
python -m pip install "vanilla-option-pricers==2.2.1"
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ArturSepp/VanillaOptionPricers/9d4ebd193923f233dc10c82ae412d32929b6d757/examples/getting_started/pricing_and_iv.py' -OutFile pricing_and_iv.py
python pricing_and_iv.py
```

Installation and fetching the script need internet access; execution is offline and writes no
files. The wheel excludes repository examples. The script prints the actual distribution
version and import path, four prices, round-trip/parity errors and separate cold/warm durations.
Allow Numba's first-call compilation to finish; measure installation separately from execution.
There is no promised time-to-first-result. Core needs NumPy and Numba; docs and hosted notebooks
are optional routes, not prerequisites.

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
Bachelier call with annualised absolute normal volatility in the same units as the
forward and strike.

Successful output reports package version and import path; input/output shapes; option codes and
prices; maximum IV and parity errors; one cold first-call duration; the mean of ten warm repeats;
and the Bachelier result. Timing values are machine-dependent and are not a benchmark. The
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
