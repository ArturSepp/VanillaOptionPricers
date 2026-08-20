
"""Show how fast one ordinary Black-Scholes-Merton call chain is priced."""

import platform
import sys
from importlib.metadata import version
from statistics import median
from time import perf_counter
from timeit import repeat as timeit_repeat

import numba
import numpy as np

import vanilla_option_pricers as vop

# This is an opt-in benchmark, even when a user explicitly passes this file to pytest.
__test__ = False

FORWARD = 100.0
TTM = 0.5
DISCOUNT_FACTOR = 0.98
CHAIN_SIZE = 61
TIMING_NUMBER = 1_000
TIMING_REPEATS = 5


def make_call_chain(size: int = CHAIN_SIZE) -> tuple[np.ndarray, ...]:
    """Create calls from strike 70 to 130 with a small volatility smile."""
    strikes = np.linspace(70.0, 130.0, size)
    moneyness = strikes / FORWARD - 1.0
    vols = 0.20 - 0.05 * moneyness + 0.10 * np.square(moneyness)
    option_types = np.full(strikes.shape, "C")
    return strikes, vols, option_types


def price_call_chain(
    strikes: np.ndarray,
    vols: np.ndarray,
    option_types: np.ndarray,
) -> np.ndarray:
    """Price every call in the chain in one compiled batch."""
    return vop.compute_bsm_vanilla_slice_prices(
        TTM,
        FORWARD,
        strikes,
        vols,
        option_types,
        DISCOUNT_FACTOR,
    )


def price_call_chain_with_numpy_vectorize(
    strikes: np.ndarray,
    vols: np.ndarray,
    option_types: np.ndarray,
) -> np.ndarray:
    """Price the same chain through the convenient NumPy vectorize wrapper."""
    return vop.compute_bsm_vanilla_price_vector(
        forward=FORWARD,
        strike=strikes,
        ttm=TTM,
        vol=vols,
        optiontype=option_types,
        discfactor=DISCOUNT_FACTOR,
    )


def median_seconds_per_call(workflow, number: int, repeats: int) -> float:
    """Return the median runtime for pricing the whole chain once."""
    totals = timeit_repeat(workflow, number=number, repeat=repeats)
    return median(total / number for total in totals)


def validate_prices(prices: np.ndarray) -> None:
    """Check that ordinary call prices are finite, positive, and strike-decreasing."""
    if not np.all(np.isfinite(prices)) or np.any(prices < 0.0):
        raise RuntimeError("The call chain contains an invalid price")
    if np.any(np.diff(prices) > 0.0):
        raise RuntimeError("Call prices should decrease as the strike increases")


def main() -> None:
    """Price one familiar option chain and explain its cold and warm timing."""
    strikes, vols, option_types = make_call_chain()

    started = perf_counter()
    prices = price_call_chain(strikes, vols, option_types)
    cold_seconds = perf_counter() - started
    validate_prices(prices)

    vectorized_prices = price_call_chain_with_numpy_vectorize(strikes, vols, option_types)
    max_price_difference = float(np.max(np.abs(prices - vectorized_prices)))
    if max_price_difference > 1e-12:
        raise RuntimeError(f"The two pricing paths disagree: {max_price_difference}")

    batch_seconds = median_seconds_per_call(
        lambda: price_call_chain(strikes, vols, option_types),
        TIMING_NUMBER,
        TIMING_REPEATS,
    )
    vectorized_seconds = median_seconds_per_call(
        lambda: price_call_chain_with_numpy_vectorize(strikes, vols, option_types),
        TIMING_NUMBER,
        TIMING_REPEATS,
    )
    prices_per_second = CHAIN_SIZE / batch_seconds
    speed_ratio = vectorized_seconds / batch_seconds
    middle = CHAIN_SIZE // 2

    print("BSM speed example: price one 61-strike call chain")
    print("Market: forward=100, maturity=6 months, discount factor=0.98")
    print("Strikes: 70 to 130; volatility is about 20% with a small smile")
    print()
    print("Three prices from the chain:")
    print(f"  strike {strikes[0]:.0f}:  call price {prices[0]:.6f}")
    print(f"  strike {strikes[middle]:.0f}: call price {prices[middle]:.6f}")
    print(f"  strike {strikes[-1]:.0f}: call price {prices[-1]:.6f}")
    print()
    print(f"First chain: {1_000.0 * cold_seconds:.3f} ms (includes Numba compilation)")
    print(
        f"Warm chain: {1_000_000.0 * batch_seconds:.3f} microseconds "
        f"for all {CHAIN_SIZE} calls"
    )
    print(f"Warm throughput: {prices_per_second:,.0f} option prices per second")
    print()
    print("Warm speed comparison using the same 61 options:")
    print(f"  Numba batch function:       {1_000_000.0 * batch_seconds:.3f} microseconds")
    print(
        "  NumPy vectorize wrapper:    "
        f"{1_000_000.0 * vectorized_seconds:.3f} microseconds"
    )
    print(f"  Numba batch is {speed_ratio:.1f}x faster on this run")
    print(f"  Maximum price difference: {max_price_difference:.3e}")
    print()
    print(
        f"Environment: {platform.platform()}, Python {sys.version.split()[0]}, "
        f"NumPy {np.__version__}, Numba {numba.__version__}, "
        f"package {version('vanilla-option-pricers')}"
    )


if __name__ == "__main__":
    main()
