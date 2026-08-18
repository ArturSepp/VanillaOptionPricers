"""
numerical tests for the Bachelier (normal) model in vanilla_option_pricers.bachelier.

Mirrors test_black_scholes.py: put-call parity, finite-difference checks of the
greeks, and an implied-vol round-trip. numpy and pytest only (no scipy). Run with
`pytest src/vanilla_option_pricers/tests/ -v`.
"""

# packages
import numpy as np
import pytest
from numba.typed import List

# vanilla_option_pricers
from vanilla_option_pricers.bachelier import (
    compute_normal_delta,
    compute_normal_delta_to_strike,
    compute_normal_price,
    compute_normal_slice_prices,
    compute_normal_slice_vegas,
    infer_normal_implied_vol,
    infer_normal_ivols_from_chain_prices,
)

# (forward, strike, ttm, vol, discfactor)
CASES = [
    (100.0, 100.0, 1.0, 20.0, 1.0),
    (100.0, 110.0, 0.5, 25.0, 0.98),
    (100.0, 90.0, 2.0, 15.0, 0.95),
    (5000.0, 5200.0, 0.5, 1000.0, 0.99),
    (5000.0, 4800.0, 1.0, 1500.0, 0.97),
    (0.04, 0.042, 2.0, 0.012, 0.99),
]


def test_normal_volatility_is_absolute_and_forward_invariant_at_the_money():
    """An ATM Bachelier price depends on absolute vol, not the forward level."""
    ttm = 0.75
    vol = 5.0
    low_forward = compute_normal_price(100.0, 100.0, ttm, vol)
    high_forward = compute_normal_price(200.0, 200.0, ttm, vol)

    assert low_forward == pytest.approx(high_forward, abs=1e-12)


@pytest.mark.parametrize("forward,strike", [(0.0, 0.0), (-0.01, 0.0)])
def test_zero_and_negative_forwards_are_supported(forward, strike):
    """Absolute normal volatility remains valid when a rate forward is non-positive."""
    vol = 0.01
    call = compute_normal_price(forward, strike, 1.0, vol, 1.0, "C")
    put = compute_normal_price(forward, strike, 1.0, vol, 1.0, "P")

    assert np.isfinite(call)
    assert np.isfinite(put)
    assert call - put == pytest.approx(forward - strike, abs=1e-10)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
def test_put_call_parity(forward, strike, ttm, vol, discfactor):
    """C - P = discfactor * (F - K)."""
    call = compute_normal_price(forward, strike, ttm, vol, discfactor, 'C')
    put = compute_normal_price(forward, strike, ttm, vol, discfactor, 'P')
    assert call - put == pytest.approx(discfactor * (forward - strike), abs=1e-6)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
@pytest.mark.parametrize("optiontype", ['C', 'P'])
def test_delta_finite_difference(forward, strike, ttm, vol, discfactor, optiontype):
    """compute_normal_delta matches d(price)/d(forward) at fixed normal vol."""
    h = 1e-4 * forward

    def price_at(fwd: float) -> float:
        return compute_normal_price(fwd, strike, ttm, vol, discfactor, optiontype)

    fd = (price_at(forward + h) - price_at(forward - h)) / (2.0 * h)
    analytic = compute_normal_delta(ttm, forward, strike, vol, optiontype, discfactor)
    assert analytic == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("forward, strike, ttm, vol, discfactor", CASES)
def test_vega_finite_difference(forward, strike, ttm, vol, discfactor):
    """discfactor * slice_vega matches d(price)/d(vol).

    For absolute normal volatility, slice_vega = n(d) * sqrt(ttm), so the discounted
    price sensitivity is discfactor * slice_vega.
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
    forward, ttm, vol = 100.0, 1.0, 20.0
    for target_delta in (0.25, 0.5, 0.75, -0.25, -0.5):
        strike = compute_normal_delta_to_strike(
            ttm=ttm, forward=forward, delta=target_delta, vol=vol
        )
        optiontype = 'C' if target_delta > 0.0 else 'P'
        recovered = compute_normal_delta(ttm, forward, strike, vol, optiontype, 1.0)
        assert recovered == pytest.approx(target_delta, abs=1e-3)


def test_chain_iv_solver_accepts_fixed_width_optiontype_arrays():
    """Rate option chains pass NumPy ``<U1`` option codes through Numba typed lists."""
    ttm = 1.0
    forward = 0.04
    vol = 0.015
    strikes = np.array([0.025, 0.035, 0.04, 0.045, 0.055])
    optiontypes = np.where(strikes >= forward, "C", "P")
    prices = compute_normal_slice_prices(
        ttm=ttm,
        forward=forward,
        strikes=strikes,
        vols=np.full_like(strikes, vol),
        optiontypes=optiontypes,
    )

    strikes_ttms = List()
    strikes_ttms.append(strikes)
    optiontypes_ttms = List()
    optiontypes_ttms.append(optiontypes)
    prices_ttms = List()
    prices_ttms.append(prices)

    inferred = infer_normal_ivols_from_chain_prices(
        ttms=np.array([ttm]),
        forwards=np.array([forward]),
        discfactors=np.array([1.0]),
        strikes_ttms=strikes_ttms,
        optiontypes_ttms=optiontypes_ttms,
        model_prices_ttms=prices_ttms,
    )

    np.testing.assert_allclose(inferred[0], vol, rtol=0.0, atol=2.0e-7)
