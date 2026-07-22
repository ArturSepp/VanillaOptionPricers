"""
numerical tests for the Bachelier (normal) model in vanilla_option_pricers.bachelier.

Mirrors tests/test_black_scholes.py: put-call parity, finite-difference checks of the
greeks, and an implied-vol round-trip. numpy and pytest only (no scipy). Run with
`pytest vanilla_option_pricers/tests/ -v`.
"""

# packages
import numpy as np
import pytest

# vanilla_option_pricers
from vanilla_option_pricers.bachelier import (
    compute_normal_price,
    compute_normal_delta,
    compute_normal_slice_vegas,
    compute_normal_delta_to_strike,
    infer_normal_implied_vol,
)

# (forward, strike, ttm, vol, discfactor)
CASES = [
    (100.0, 100.0, 1.0, 0.20, 1.0),
    (100.0, 110.0, 0.5, 0.25, 0.98),
    (100.0, 90.0, 2.0, 0.15, 0.95),
    (5000.0, 5200.0, 0.5, 0.20, 0.99),
    (5000.0, 4800.0, 1.0, 0.30, 0.97),
]


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
def test_put_call_parity(forward, strike, ttm, vol, discfactor):
    """C - P = discfactor * (F - K)."""
    call = compute_normal_price(forward, strike, ttm, vol, discfactor, 'C')
    put = compute_normal_price(forward, strike, ttm, vol, discfactor, 'P')
    assert call - put == pytest.approx(discfactor * (forward - strike), abs=1e-6)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
@pytest.mark.parametrize("optiontype", ['C', 'P'])
def test_delta_finite_difference(forward, strike, ttm, vol, discfactor, optiontype):
    """compute_normal_delta matches d(price)/d(forward) at fixed absolute normal vol.

    `vol` is a *relative* normal vol (sdev = forward*vol*sqrt(ttm)), so the finite
    difference must hold the absolute vol sigma_N = forward*vol fixed while bumping
    forward, otherwise it also perturbs sigma_N.
    """
    sigma_n = forward * vol
    h = 1e-4 * forward

    def price_at(fwd: float) -> float:
        return compute_normal_price(fwd, strike, ttm, sigma_n / fwd, discfactor, optiontype)

    fd = (price_at(forward + h) - price_at(forward - h)) / (2.0 * h)
    analytic = compute_normal_delta(ttm, forward, strike, vol, optiontype, discfactor)
    assert analytic == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
def test_vega_finite_difference(forward, strike, ttm, vol, discfactor):
    """discfactor * slice_vega matches d(price)/d(vol).

    slice_vega = forward * n(d) * sqrt(ttm) already carries the forward factor, so the
    discounted price sensitivity is discfactor * slice_vega (not * forward again).
    """
    h = 1e-5
    up = compute_normal_price(forward, strike, ttm, vol + h, discfactor, 'C')
    dn = compute_normal_price(forward, strike, ttm, vol - h, discfactor, 'C')
    fd = (up - dn) / (2.0 * h)
    slice_vega = compute_normal_slice_vegas(ttm, forward, np.array([strike]), np.array([vol]))[0]
    analytic = discfactor * slice_vega
    assert analytic == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
@pytest.mark.parametrize("optiontype", ['C', 'P'])
def test_implied_vol_roundtrip(forward, strike, ttm, vol, discfactor, optiontype):
    """price -> infer_normal_implied_vol recovers the input vol (identifiable regime)."""
    price = compute_normal_price(forward, strike, ttm, vol, discfactor, optiontype)
    iv = infer_normal_implied_vol(forward=forward, ttm=ttm, strike=strike,
                                  given_price=price, discfactor=discfactor, optiontype=optiontype)
    assert iv == pytest.approx(vol, abs=1e-6)


def test_implied_vol_out_of_range_is_nan():
    """an unreachable price returns nan rather than a silent bracket bound."""
    iv = infer_normal_implied_vol(forward=100.0, ttm=1.0, strike=100.0,
                                  given_price=1.0e6, discfactor=1.0, optiontype='C')
    assert np.isnan(iv)


def test_delta_to_strike_roundtrip():
    """compute_normal_delta_to_strike inverts compute_normal_delta (ncdf_inv-limited)."""
    forward, ttm, vol = 100.0, 1.0, 0.2
    for target_delta in (0.25, 0.5, 0.75, -0.25, -0.5):
        strike = compute_normal_delta_to_strike(ttm=ttm, forward=forward, delta=target_delta, vol=vol)
        optiontype = 'C' if target_delta > 0.0 else 'P'
        recovered = compute_normal_delta(ttm, forward, strike, vol, optiontype, 1.0)
        assert recovered == pytest.approx(target_delta, abs=1e-3)
