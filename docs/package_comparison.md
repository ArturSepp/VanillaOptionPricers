---
myst:
  html_meta:
    description: >-
      Compare vanilla-option-pricers with QuantLib-Python, vollib, py-vollib-vectorized, and
      fast-vollib by model scope, array behavior, dependencies, and intended workflow.
---

# Choosing a vanilla-option pricing package

There is no universal winner in this comparison. A two-dependency array kernel, a scalar formula
library, a pandas-oriented compatibility layer, a multi-backend tensor library, and a full
derivatives framework solve different problems.

This guide compares documented public capabilities, not speed. It does not install the other
packages or run a performance race. “Not documented” means that the cited public scope did not
claim the feature; it is not proof that a custom implementation is impossible.

## Release snapshot

Audit date: **2026-08-16**. Versions and release dates come from each distribution's official
PyPI record.

| Distribution | Stable version | Latest release in the audited record | Maintenance signal used here |
|---|---:|---:|---|
| [`vanilla-option-pricers`](https://pypi.org/project/vanilla-option-pricers/) | 1.3.0 | 2026-07-22 | Current package under comparison. |
| [`QuantLib`](https://pypi.org/project/QuantLib/) | 1.43 | 2026-07-14 | 2026 release; PyPI classifies the project as mature. |
| [`vollib`](https://pypi.org/project/vollib/) | 1.0.11 | 2026-06-01 | Canonical distribution and namespace; several 2026 releases. |
| [`py-vollib-vectorized`](https://pypi.org/project/py-vollib-vectorized/) | 0.1.1 | 2021-02-28 | Historical comparison; no newer PyPI release was present. |
| [`fast-vollib`](https://pypi.org/project/fast-vollib/) | 0.1.7 | 2026-07-03 | Emerging project with releases from March through July 2026. |

Release recency is context, not a quality score. Recheck it before making a long-lived dependency
decision.

## Model and valuation surface

The source keys in every competitor cell resolve to official package metadata, project
documentation, repository files, or API references listed under [Primary sources](#primary-sources).

| Capability | `vanilla-option-pricers` | QuantLib-Python | `vollib` | `py-vollib-vectorized` | `fast-vollib` |
|---|---|---|---|---|---|
| Formula and instrument scope | European forward-based BSM and Bachelier kernels. | Broad quantitative-finance framework with instruments, processes, term structures, and replaceable pricing engines. [Q1, Q2] | Black-76, Black-Scholes, and Black-Scholes-Merton formulae. [V1] | Vectorized Black-76, Black-Scholes, and Black-Scholes-Merton functions built on `py_vollib`. [PV1, PV2] | Black-76, Black-Scholes, and Black-Scholes-Merton pricing, IV, Greeks, plus an IV-surface evaluation harness. [F1] |
| BSM and Black-76 | BSM functions consume a caller-supplied forward and discount factor. | Both Black-76 formula utilities and Black-Scholes-Merton processes are documented. [Q3, Q4] | Separate modules for Black, Black-Scholes, and BSM. [V1] | Vectorizes the corresponding `py_vollib` model families. [PV1] | Three explicit pricing entry points cover Black-76, Black-Scholes, and BSM. [F1] |
| Bachelier | Yes. `vol` is annualised **absolute normal volatility** in forward/strike units. See [unit convention](bachelier_convention.md). | Yes. The formula API consumes an **absolute maturity standard deviation**, `absolute_vol * sqrt(T)`, and provides Bachelier IV routines. [Q3] | Not in the documented model list. [V1] | Not in the documented model list. [PV1, PV2] | Not in the documented model list. [F1] |
| Greeks | Selected BSM and normal Greeks; exact coverage varies by helper. See [API inventory](api.md). | `VanillaOption` exposes delta, forward delta, gamma, theta, vega, rho, and other sensitivities when its engine supplies them. [Q5] | Analytical and numerical Greeks for the three documented formula families. [V1] | Vectorized delta, gamma, theta, rho, vega, and an all-Greeks helper. [PV1] | Delta, gamma, theta, rho, vega, and an all-Greeks helper. [F1] |
| Implied volatility | Safeguarded Newton steps with bisection fallback for BSM and absolute Bachelier; scalar, slice, and chain helpers. See [IV limits](numerical_accuracy_and_performance.md#implied-volatility-solver-limits). | Instrument-level BSM inversion plus several Black-76 and Bachelier formula routines; there is no single package-wide IV method. [Q3, Q5] | Peter Jaeckel's LetsBeRational method for the three formula families. [V1] | Built on `py_vollib`/LetsBeRational with vectorized, Numba-oriented wrappers. [PV1, PV2] | Vectorized Halley iteration with a compiled bisection fallback. [F1] |
| Inverse contracts | `IC`/`IP` select branches only; the caller must normalize the coin quote and compute the inverse hedge. See [inverse normalization](inverse_options.md). | Native inverse-contract support was not assessed; it is not claimed by the cited vanilla-option pages. [Q5] | Not documented in the cited public model scope. [V1] | Not documented in the cited public model scope. [PV1, PV2] | Not documented in the cited public model scope. [F1] |
| Calendars, curves, and instruments | Intentionally absent; callers supply forward, discount factor, and maturity. | Core scope: calendars, term structures, instruments, processes, and pricing engines. [Q1, Q2, Q6] | Not part of the documented formula-library scope. [V1] | Not part of the documented vectorized formula scope. [PV1] | Not part of the documented pricing/IV/Greeks scope. [F1] |

## Inputs, dependencies, and execution

| Capability | `vanilla-option-pricers` | QuantLib-Python | `vollib` | `py-vollib-vectorized` | `fast-vollib` |
|---|---|---|---|---|---|
| Scalar and batch inputs | Scalar Numba dispatchers, five NumPy-broadcasting convenience wrappers, and explicit slice/grid/chain helpers. | Python bindings expose C++ objects and scalar-valued formula/instrument APIs. A standardized NumPy batch API was not assessed. [Q2, Q3, Q5] | Documented examples use scalar formula calls. A native batch contract is not documented. [V1] | Accepts scalars, tuples, lists, NumPy arrays, pandas Series, and one-column DataFrames. [PV1] | Vectorized scalar/array calls and a DataFrame helper; examples mix scalar and array inputs. [F1] |
| Broadcasting and alignment | Only the five `*_vector` wrappers promise NumPy broadcasting. Slice/grid/chain helpers require their documented layouts. See [shape contract](array_shapes_and_numba.md#shape-contract). | NumPy broadcasting or alignment semantics are not documented by the cited bindings/API pages. [Q2, Q3] | Broadcasting semantics are not documented by the cited project page. [V1] | Automatic broadcasting is documented after importing the monkey patch; dedicated utility functions are also available. [PV1] | The quick start demonstrates scalar/array batch inputs; a complete cross-backend broadcasting contract was not found, so unusual shapes should be tested. [F1] |
| Core numerical/data dependencies | NumPy and Numba only. | Compiled QuantLib C++ Python wheels; the cited package page does not present a pandas/SciPy array layer. [Q1, Q2] | Current project documentation lists NumPy, pandas, SciPy, and the LetsBeRational support packages. [V1] | Requires `py_vollib`, Numba, NumPy, pandas, SciPy, and `py_lets_be_rational`. [PV2] | Default dependencies include NumPy, SciPy, pandas, joblib, and psutil; PyTorch, JAX, and Numba are optional extras. [F2] |
| CPU and GPU backends | CPU through NumPy/Numba; no GPU backend. | Compiled C++ CPU library exported to Python; no GPU backend is claimed on the cited project surfaces. [Q1, Q2] | CPU-oriented Python implementation; no GPU backend is documented. [V1] | Numba CPU speedups are documented; no GPU backend is documented. [PV1] | NumPy default, PyTorch with CUDA, and JAX with JIT; automatic selection prefers CUDA, then JAX, then NumPy. [F1] |
| JIT cold start | First call for each new Numba signature compiles; the five `numpy.vectorize` wrappers remain Python convenience loops. See [first-call compilation](array_shapes_and_numba.md#first-call-compilation). | No Python JIT contract is documented; installation uses prebuilt compiled bindings where a wheel is available. [Q1, Q2] | No JIT contract is documented. [V1] | Numba is required for its speedups, so first-use compilation is relevant; the project does not publish a complete cold-start contract. [PV1, PV2] | Backend-dependent: JAX is explicitly JIT-based; NumPy is the default, and Numba is an optional extra. [F1, F2] |
| Intended fit | Small, explicit array kernels inside calibration, surface-fitting, or research pipelines where the caller owns market conventions. | Applications needing dates, calendars, curves, instruments, processes, and interchangeable engines. [Q1, Q2, Q6] | Focused scalar Black/BS/BSM price, IV, and Greek calculations. [V1] | Existing NumPy/pandas workflows that want the historical `py_vollib` monkey-patch or utility API. [PV1] | Batch/tensor work that benefits from NumPy, PyTorch/CUDA, JAX, DataFrames, or differentiable surface checks. [F1, F2] |

## Workflow-based choice guide

Choose **QuantLib-Python** when a vanilla option is one instrument inside a dated market model:
you need calendars, day counts, bootstrapped curves, volatility term structures, exercise objects,
and interchangeable analytic, finite-difference, or Monte Carlo engines. That framework scope is
the feature, even when it is more machinery than a single formula call requires. [Q1, Q2, Q6]

Choose **`vollib`** when the task is a focused scalar Black-76, Black-Scholes, or BSM calculation
and you specifically want the canonical package's LetsBeRational IV path and analytical or
numerical Greeks. [V1]

Choose **`py-vollib-vectorized`** when maintaining an existing workflow that already relies on its
`py_vollib` monkey patch, pandas inputs, return-format conventions, or dedicated vectorized
utilities. Its last audited PyPI release is from 2021, so validate it against your supported
Python/NumPy/Numba stack before adopting it in a new long-lived environment. [PV1, PV2, PV3]

Choose **`fast-vollib`** when arrays or tensors must move between NumPy, PyTorch/CUDA, and JAX, or
when a pandas helper and differentiable IV-surface diagnostics belong in the same package. Its
0.x version and recent release history are reasons to pin and test the exact API, not reasons to
reject or prefer it automatically. [F1, F2, F3]

Choose **`vanilla-option-pricers`** when the inputs are already normalized forwards, discount
factors, maturities, strikes, and option codes; BSM and relative-Bachelier kernels are enough; and
keeping the runtime surface to NumPy plus Numba matters. Its explicit slice/grid/chain contracts
can suit calibration loops, but it is the wrong choice if you need calendars, curves, pandas,
GPU tensors, or a general instrument object model.

It is also reasonable to use more than one package: for example, build curves and schedules in a
framework and pass audited forward/discount inputs to a small array kernel. The integration layer
must own unit, discounting, day-count, and payoff-convention checks.

## Comparison limits

- This is a documentation audit, not an exhaustive source-code or numerical audit of the other
  projects.
- No competitor was installed, timed, or used as a test oracle.
- “Not documented” cells avoid treating the absence of a feature from a public overview as proof
  of impossibility.
- Backend behavior, optional extras, and compatibility can change between releases. Pin versions
  and test the shapes, dtypes, devices, and boundary cases used by your application.
- Adoption counters are deliberately omitted because they do not determine technical fit.

## Primary sources

- **Q1:** [QuantLib 1.43 on PyPI](https://pypi.org/project/QuantLib/) — version, release date,
  maturity classifier, compiled wheel distribution, and project scope.
- **Q2:** [QuantLib project](https://www.quantlib.org/) and
  [official bindings repository](https://github.com/lballabio/QuantLib-SWIG) — framework and
  Python-binding scope.
- **Q3:** [QuantLib Black/Bachelier formula API](https://rkapl123.github.io/QLAnnotatedSource/d5/d2f/blackformula_8hpp.html)
  — Black-76, standard-deviation inputs, IV routines, and Bachelier functions.
- **Q4:** [QuantLib Black-Scholes-Merton process API](https://rkapl123.github.io/QLAnnotatedSource/dc/dbb/class_quant_lib_1_1_black_scholes_merton_process.html).
- **Q5:** [QuantLib `VanillaOption` API](https://rkapl123.github.io/QLAnnotatedSource/d2/d47/class_quant_lib_1_1_vanilla_option.html)
  — instrument Greeks and implied volatility.
- **Q6:** [QuantLib `TermStructure` API](https://rkapl123.github.io/QLAnnotatedSource/d7/dbb/class_quant_lib_1_1_term_structure.html)
  and [pricing-engine design](https://www.quantlib.org/quep/quep005.html).
- **V1:** [`vollib` official repository](https://github.com/vollib/py_vollib) and
  [PyPI record](https://pypi.org/project/vollib/) — canonical namespace, formula families,
  Greeks, LetsBeRational IV, dependencies, version, and release date.
- **PV1:** [`py_vollib_vectorized` official repository](https://github.com/marcdemers/py_vollib_vectorized)
  — inputs, broadcasting, Numba behavior, model families, Greeks, and DataFrame helper.
- **PV2:** [`py_vollib_vectorized` package metadata](https://github.com/marcdemers/py_vollib_vectorized/blob/main/setup.py)
  — dependencies and declared scope.
- **PV3:** [`py-vollib-vectorized` PyPI record](https://pypi.org/project/py-vollib-vectorized/)
  — version and release date.
- **F1:** [`fast-vollib` official repository](https://github.com/raeidsaqur/fast-vollib) — model
  families, IV method, Greeks, array/DataFrame API, surface harness, and backend selection.
- **F2:** [`fast-vollib` package metadata](https://github.com/raeidsaqur/fast-vollib/blob/main/pyproject.toml)
  — default dependencies and optional backends/extras.
- **F3:** [`fast-vollib` PyPI record](https://pypi.org/project/fast-vollib/) — version and release
  history.
