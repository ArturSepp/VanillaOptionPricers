"""Independent price and IV checks, including tails and cancellation boundaries."""

import math

import numpy as np
import pytest
from numba import njit

from vanilla_option_pricers.black_scholes import (
    compute_bsm_vanilla_price,
    infer_bsm_implied_vol,
)
from vanilla_option_pricers.bachelier import compute_normal_price, infer_normal_implied_vol
from vanilla_option_pricers.tests._numerical_reference import REFERENCE_CASES


@pytest.mark.parametrize("case", REFERENCE_CASES)
def test_independent_prices(case):
    """Literal 110-digit references expose shared errors hidden by round trips."""
    model, f, k, t, vol, df, q, price, _ = case
    function = compute_bsm_vanilla_price if model == "black" else compute_normal_price
    actual = function(forward=f, strike=k, ttm=t, vol=vol, discfactor=df, optiontype=q)
    assert actual >= 0
    assert actual == pytest.approx(price, rel=3e-13, abs=0.0)


@pytest.mark.parametrize("case", [c for c in REFERENCE_CASES if c[-1] is not None])
def test_independent_implied_volatilities(case):
    """Invert independently rounded prices, with independently computed IV targets."""
    model, f, k, t, _, df, q, price, iv = case
    function = infer_bsm_implied_vol if model == "black" else infer_normal_implied_vol
    actual = function(
        forward=f,
        strike=k,
        ttm=t,
        given_price=price,
        discfactor=df,
        optiontype=q,
        vol_lower=1e-12,
        vol_upper=100.0,
        tol=1e-13,
    )
    assert actual == pytest.approx(iv, rel=5e-12, abs=2e-13)


@pytest.mark.parametrize("q", ["C", "P", "IC", "IP"])
@pytest.mark.parametrize("vol", [0.005, 6.0])
@pytest.mark.parametrize("nan_bounds", [True, False])
def test_black_preserves_custom_bounds(q, vol, nan_bounds):
    """The unbounded internal inverse must respect the public 1%-500% bracket."""
    price = 100 * math.erf(vol / math.sqrt(8.0))
    result = infer_bsm_implied_vol(
        100.0, 1.0, 100.0, price, optiontype=q, is_bounds_to_nan=nan_bounds
    )
    if nan_bounds:
        assert math.isnan(result)
    else:
        assert result == (0.01 if vol < 0.01 else 5.0)
    assert infer_bsm_implied_vol(
        100.0, 1.0, 100.0, price, optiontype=q, vol_lower=0.001, vol_upper=8.0
    ) == pytest.approx(vol, rel=1e-13)


@pytest.mark.parametrize("function", [infer_bsm_implied_vol, infer_normal_implied_vol])
@pytest.mark.parametrize("price", [math.nan, -1.0, 0.0])
def test_price_sentinels(function, price):
    """Missing/nonpositive prices retain the NaN or lower-bound contract."""
    assert math.isnan(function(100.0, 1.0, 100.0, price))
    assert function(100.0, 1.0, 100.0, price, vol_lower=0.02, is_bounds_to_nan=False) == 0.02


@pytest.mark.parametrize("function", [compute_bsm_vanilla_price, compute_normal_price])
@pytest.mark.parametrize("ttm,vol", [(0.0, 0.2), (1.0, 0.0)])
def test_discounted_intrinsic_limits(function, ttm, vol):
    """The stable OTM decomposition has the correct zero-diffusion limit."""
    assert function(120.0, 100.0, ttm, vol, optiontype="C", discfactor=0.9) == 18.0
    assert function(100.0, 120.0, ttm, vol, optiontype="P", discfactor=0.9) == 18.0


@njit
def _string_batch(forwards, strikes, prices, codes, normal):
    result = np.empty(prices.size)
    for i in range(prices.size):
        if normal:
            result[i] = infer_normal_implied_vol(
                forwards[i], 1.0, strikes[i], prices[i], optiontype=codes[i]
            )
        else:
            result[i] = infer_bsm_implied_vol(
                forwards[i], 1.0, strikes[i], prices[i], optiontype=codes[i]
            )
    return result


@pytest.mark.parametrize("normal", [True, False])
def test_compiled_fixed_width_strings_include_itm_and_inverse(normal):
    """Parity conversion must work in nopython loops with <U2 option-code arrays."""
    f = np.array([120.0, 80.0, 120.0, 80.0])
    k = np.full(4, 100.0)
    codes = np.array(["C", "P", "IC", "IP"])
    function = compute_normal_price if normal else compute_bsm_vanilla_price
    vol = 20.0 if normal else 0.2
    prices = np.array([function(f[i], k[i], 1.0, vol, optiontype=codes[i]) for i in range(4)])
    np.testing.assert_allclose(_string_batch(f, k, prices, codes, normal), vol, rtol=2e-12)


def test_iteration_limit_and_tolerance_are_active():
    """Custom solver controls remain usable rather than silently ignored."""
    kwargs = dict(forward=100.0, strike=120.0, ttm=1.0, given_price=2.147298810578146)
    zero = infer_bsm_implied_vol(**kwargs, max_iters=0)
    assert zero == 0.5 * (0.01 + 5.0)
    one = infer_bsm_implied_vol(**kwargs, max_iters=1)
    full = infer_bsm_implied_vol(**kwargs, max_iters=100, tol=1e-13)
    assert abs(full - 0.2) < 1e-13
    assert abs(full - 0.2) <= abs(one - 0.2) + 1e-16
