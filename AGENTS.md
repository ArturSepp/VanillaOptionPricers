## Python environment (mandatory)

- Never create, use, or install packages into a Python virtual environment anywhere under `C:\Users\artur\OneDrive`.
- Keep this repository's environment outside OneDrive at `C:\Python\VanillaOptionPricers312`.
- Use `C:\Python\VanillaOptionPricers312\Scripts\python.exe` for Python, tests, linters, and package installation.
- If it is missing, create it with `py -3.12 -m venv C:\Python\VanillaOptionPricers312`.
- Never run plain `uv sync` or plain `uv run` from this checkout: uv otherwise creates `<repo>\.venv` even when uv was launched through a Python executable under `C:\Python`.
- If a uv project operation is required, first set `UV_PROJECT_ENVIRONMENT=C:\Python\VanillaOptionPricers312`; for pip-style operations prefer `uv pip ... --python C:\Python\VanillaOptionPricers312\Scripts\python.exe`.
- If any OneDrive-local environment already exists, do not use it; report it for removal.

# AGENTS.md

Guidance for AI coding agents working in the **VanillaOptionPricers** repository.

## Project overview

`vanilla-option-pricers` provides fast, vectorised pricers and implied volatility
fitters for vanilla options under the Black-Scholes-Merton and Bachelier (normal)
models. It also exposes `IC`/`IP` branches used by coin-denominated inverse-option
workflows; callers own the required measure and payoff normalisation. Scalar kernels and
slice/grid/chain helpers are `numba`-compiled, while five convenience wrappers use
`numpy.vectorize`. The only runtime dependencies are numpy and numba.

The design goal is minimalism and speed inside calibration loops and surface fitters —
it is deliberately not a derivatives framework. Distribution name
`vanilla-option-pricers`; import name `vanilla_option_pricers`. Licensed MIT
(`LICENSE.txt`).

## Ecosystem position

This package is one of ten public Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Check the owning package before
adding a capability or copying code between repositories.

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | performance analytics, backtesting, and factsheet reporting |
| `optimalportfolios` | OptimalPortfolios | portfolio construction and rolling backtesting |
| `factorlasso` | FactorLasso | sparse factor-model estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data in pandas DataFrames |
| `stochvolmodels` | StochVolModels | stochastic-volatility pricing and calibration |
| `trendfollowing` | TrendFollowingSystems | closed-form trend-following analytics |
| `privateassets` | PrivateAssets | multi-factor PME for private assets |
| `goal-based-allocation` | GoalBasedAllocation | goal-based allocation under regime-switching jump-diffusions |
| `vanilla-option-pricers` | VanillaOptionPricers | Numba-vectorised BSM and Bachelier pricing |
| `option-chain-analytics` | OptionChainAnalytics | point-in-time option-chain data and queries |

Core dependency edges: `optimalportfolios` consumes `qis` and `factorlasso`;
`trendfollowing` and `privateassets` consume `qis`; `stochvolmodels` consumes
`vanilla-option-pricers`; `option-chain-analytics` consumes `qis` and
`vanilla-option-pricers`. The remaining packages have no core stack dependencies.

Optional edges: PrivateAssets' `factors` extra adds `factorlasso`; StochVolModels'
`research` extra adds `qis` and `option-chain-analytics`; OCA's `bloomberg` and `all`
extras add `bbg-fetch`. Core imports must work without optional dependencies.
OCA never imports StochVolModels or the private SigmaStrats consumer. Exact
maintainer-tool exceptions are recorded in `.github/stack-policy.json`; they do
not authorise adding those dependencies to core or importing them at package root.

## Repository layout

```
src/vanilla_option_pricers/
  __init__.py              package-root public exports
  black_scholes.py         Black-Scholes-Merton pricers and implied volatility fitters
  bachelier.py             Bachelier normal pricers and implied volatility fitters
  utils.py                 shared numerical helpers
  tests/
    test_black_scholes.py  parity, reference-value, and finite-difference tests
    test_bachelier.py      parity, finite-difference, and implied-volatility tests
examples/
  README.md                example scope and execution guidance
  performance/
    bsm_speed.py           manual timing diagnostic; not a CI correctness gate
```

## Commands

```bash
uv sync --locked --group test
uv run --no-sync pytest                              # correctness checks, as CI runs them
python examples/performance/bsm_speed.py             # optional local timing diagnostic
uv run --locked --only-group lint ruff check src/vanilla_option_pricers/ examples/
```

Pytest's configured `testpaths` points at `src/vanilla_option_pricers/tests/`. Supported Python
is >= 3.10. CI runs the correctness suite on Python 3.10 - 3.14 under Ubuntu and Python 3.12
under Windows and macOS. Root `examples/` is repository-only and excluded from the wheel.

## Conventions

- Line length 100 (`ruff`, rules `E`, `F`, `W`).
- `pyproject.toml` contains narrow per-file waivers for pre-existing lint debt in the unchanged
  numerical modules and package re-exports. Do not expand those waivers for new code.
- Scalar pricing functions are Numba dispatchers. Explicit slice, grid, and chain helpers use
  compiled loops over aligned arrays; five `*_vector` conveniences use `numpy.vectorize`.
- Option type selection is by the string codes `C`, `P`, `IC`, and `IP`; there is no exported
  option-type enum.
- Do not promise arbitrary broadcasting. Scalar functions, `numpy.vectorize` wrappers, aligned
  slice/grid arrays, and per-expiry chain containers are distinct public paths.
- There is no object model, calendar handling, or pandas integration.
- New functionality should come with a numerical check against a reference value or
  against put-call parity.

## Constraints — do not do these

- Do not add runtime dependencies. numpy and numba only — in particular, do not import
  `scipy` (for example for root finding in the implied volatility fitters) or `pandas`.
  The two-dependency footprint is the reason this package exists.
- Do not add American, exotic, or path-dependent payoffs, term structures, settlement
  conventions, or stochastic volatility. Those belong in `stochvolmodels`.
- Do not introduce Python-level loops over strikes or expiries in place of vectorised
  operations.
- Do not wrap the functions in classes; the API is deliberately function-based.

## Repository-specific agent artefacts

By maintainer direction, all VanillaOptionPricers roadmaps, execution plans, audits, and
reports live in the ignored `agents/` directory. This repository-specific rule overrides the
generic roadmap location inside the generated shared-agent block below; do not edit that
generated block directly.

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-09-08, agent core v1.5 -->

## Domain invariants

- Conventions are stated, never implied: volatility quotation, rate and dividend conventions,
  annualisation. One convention per concept across the stack — if this package and a sibling
  disagree, that is a bug to report, not a difference to accommodate.

## Dependency surface

This package is standalone: it imports nothing from the stack, and its two-dependency runtime
surface — numpy, numba — is the reason it exists. Ask before adding any dependency.

**Never invent a symbol.** If a function, class, or keyword argument is not in the export
surface of this package or of a dependency, it does not exist. Check in one line —
`python -c "import vanilla_option_pricers as v; print([n for n in dir(v) if not n.startswith('_')])"`
— and say a symbol is missing rather than producing code that calls it.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: pricing formulas, implied-volatility inversion, greeks. Put-call parity and reference
  values are the checks computed a different way — verify against them and say so.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Roadmap execution

Feature roadmaps live at the repository root as `ROADMAP_<feature>.md`. An execution request
names the file and the stage. A stage is complete when its stated verification command passes;
its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

For an authorized publication: commit, tag that exact main-reachable commit as
`v<version>`, then build, verify and publish its artifacts. Frequent PyPI updates are
supported. A GitHub Release page is optional and created only when requested; it is not
required for a local build, pip installation or routine package publication. Development
versions on main may be ahead of PyPI. Do not publish or bump a version for unrelated work.

## Known issues

No Python 3.14 compatibility exclusion is active. The installed package uses the standard
`src/vanilla_option_pricers/` layout; repository-only examples live under root `examples/`.
