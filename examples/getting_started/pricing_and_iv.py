"""Offline first-success workflow for vanilla-option-pricers."""

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from numba.typed import List

import vanilla_option_pricers as vop

forward = 100.0
discfactor = 0.98
ttm = 0.50
strikes = np.array([90.0, 100.0, 100.0, 110.0])
option_types = np.array(["P", "P", "C", "C"])
solve_types = List(option_types.tolist())
lognormal_vols = np.full(strikes.shape, 0.20)


def run_bsm_workflow() -> tuple[np.ndarray, np.ndarray]:
    """Price one aligned BSM slice and recover its implied volatilities."""
    prices = vop.compute_bsm_vanilla_slice_prices(
        ttm,
        forward,
        strikes,
        lognormal_vols,
        option_types,
        discfactor,
    )
    implied_vols = vop.infer_bsm_ivols_from_slice_prices(
        ttm,
        forward,
        discfactor,
        strikes,
        solve_types,
        prices,
    )
    return prices, implied_vols


cold_started = perf_counter()
prices, implied_vols = run_bsm_workflow()
cold_seconds = perf_counter() - cold_started

warm_repetitions = 10
warm_started = perf_counter()
for _ in range(warm_repetitions):
    warm_prices, warm_implied_vols = run_bsm_workflow()
warm_mean_seconds = (perf_counter() - warm_started) / warm_repetitions

max_iv_error = float(np.max(np.abs(implied_vols - lognormal_vols)))
parity_error = float(prices[2] - prices[1] - discfactor * (forward - strikes[1]))

absolute_normal_vol = 5.0
relative_normal_vol = absolute_normal_vol / forward
normal_strike = 102.0
normal_price = vop.compute_normal_price(
    forward,
    normal_strike,
    ttm,
    relative_normal_vol,
    discfactor,
    "C",
)
normal_implied_vol = vop.infer_normal_implied_vol(
    forward,
    ttm,
    normal_strike,
    normal_price,
    discfactor,
    "C",
)

if max_iv_error > 1e-7:
    raise RuntimeError(f"BSM implied-volatility round trip failed: {max_iv_error}")
if abs(parity_error) > 1e-10:
    raise RuntimeError(f"BSM put-call parity failed: {parity_error}")
if abs(normal_implied_vol - relative_normal_vol) > 1e-7:
    raise RuntimeError("Bachelier implied-volatility round trip failed")
if not (
    np.array_equal(prices, warm_prices)
    and np.array_equal(implied_vols, warm_implied_vols)
):
    raise RuntimeError("Warm BSM workflow changed the deterministic result")

print(f"distribution_version={version('vanilla-option-pricers')}")
print(f"import_path={Path(vop.__file__).resolve()}")
print(
    "array_shapes="
    f"strikes{strikes.shape} option_types{option_types.shape} "
    f"prices{prices.shape} implied_vols{implied_vols.shape}"
)
print("option_types=" + str(option_types.tolist()))
print("prices=" + np.array2string(prices, precision=8))
print(f"max_iv_error={max_iv_error:.3e} parity_error={parity_error:.3e}")
print(
    f"cold_first_call_seconds={cold_seconds:.6f} "
    f"warm_mean_seconds={warm_mean_seconds:.6f} "
    f"warm_repetitions={warm_repetitions}"
)
print(
    f"bachelier_relative_vol={relative_normal_vol:.6f} "
    f"absolute_normal_vol={absolute_normal_vol:.6f} "
    f"price={normal_price:.8f} iv_error="
    f"{normal_implied_vol - relative_normal_vol:.3e}"
)
print("bsm_convention=forward, discount factor, years, annualised lognormal volatility")
print("bachelier_convention=relative vol; absolute_normal_vol = forward * relative_vol")
print(
    "change_first=forward, discfactor, ttm, strikes, option_types, model_convention"
)
