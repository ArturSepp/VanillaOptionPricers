"""
Numerical checks for the Black-Scholes-Merton pricer and its greeks.

Each greek is compared against a finite difference of the pricer itself, so the
pricer is the reference. Theta is taken with the spot held fixed (the forward and
the discount factor both move with time to maturity), matching the discounting the
analytic theta assumes.
"""
import math

import numpy as np

from vanilla_option_pricers.black_scholes import (
    compute_bsm_vanilla_delta,
    compute_bsm_vanilla_gamma,
    compute_bsm_vanilla_price,
    compute_bsm_vanilla_theta,
    compute_bsm_vanilla_vega,
)

# (spot, strike, ttm, vol, rate)
CASES = [
    (100.0, 100.0, 1.00, 0.20, 0.00),
    (100.0, 100.0, 1.00, 0.20, 0.05),
    (90.0, 100.0, 2.00, 0.30, 0.10),
    (110.0, 95.0, 0.50, 0.25, 0.03),
    (100.0, 120.0, 0.25, 0.40, 0.07),
    (50.0, 65.0, 0.43, 0.31, 0.02),
]


def _price(spot, strike, ttm, vol, optiontype, rate):
    return compute_bsm_vanilla_price(spot * np.exp(rate * ttm), strike, ttm, vol,
                                     optiontype, np.exp(-rate * ttm))


def test_price_put_call_parity():
    for spot, strike, ttm, vol, rate in CASES:
        call = _price(spot, strike, ttm, vol, 'C', rate)
        put = _price(spot, strike, ttm, vol, 'P', rate)
        forward, discfactor = spot * np.exp(rate * ttm), np.exp(-rate * ttm)
        assert abs((call - put) - discfactor * (forward - strike)) < 1e-10


def test_theta_reference_value():
    # ATM, zero rate: closed-form Black decay -F n(d1) vol / (2 sqrt(T)), computed here
    # independently of the library.
    forward, strike, ttm, vol = 100.0, 100.0, 1.0, 0.20
    d1 = 0.5 * vol * math.sqrt(ttm)
    n_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    expected = -forward * n_d1 * vol / (2.0 * math.sqrt(ttm))
    theta = compute_bsm_vanilla_theta(ttm, forward, strike, vol, 'C', 1.0, 0.0)
    assert abs(theta - expected) < 1e-6


def test_theta_matches_finite_difference():
    h = 1e-6
    for spot, strike, ttm, vol, rate in CASES:
        for optiontype in ('C', 'P'):
            forward, discfactor = spot * np.exp(rate * ttm), np.exp(-rate * ttm)
            analytic = compute_bsm_vanilla_theta(ttm, forward, strike, vol, optiontype,
                                                 discfactor, rate)
            fd = -(_price(spot, strike, ttm + h, vol, optiontype, rate)
                   - _price(spot, strike, ttm - h, vol, optiontype, rate)) / (2.0 * h)
            assert abs(analytic - fd) < 1e-4


def test_delta_vega_gamma_match_finite_difference():
    # forward-measure greeks (discfactor = 1) against a bump of the pricer; a second
    # difference needs a wider step for gamma.
    for forward, strike, ttm, vol, _ in CASES:
        h = 1e-4
        vega = compute_bsm_vanilla_vega(ttm, forward, strike, vol)
        vega_fd = (compute_bsm_vanilla_price(forward, strike, ttm, vol + h, 'C')
                   - compute_bsm_vanilla_price(forward, strike, ttm, vol - h, 'C')) / (2.0 * h)
        assert abs(vega - vega_fd) < 1e-3

        gamma = compute_bsm_vanilla_gamma(ttm, forward, strike, vol)
        hg = 1e-2
        gamma_fd = (compute_bsm_vanilla_price(forward + hg, strike, ttm, vol, 'C')
                    - 2.0 * compute_bsm_vanilla_price(forward, strike, ttm, vol, 'C')
                    + compute_bsm_vanilla_price(forward - hg, strike, ttm, vol, 'C')) / hg ** 2
        assert abs(gamma - gamma_fd) < 1e-3

        for optiontype in ('C', 'P'):
            delta = compute_bsm_vanilla_delta(ttm, forward, strike, vol, optiontype)
            up = compute_bsm_vanilla_price(forward + h, strike, ttm, vol, optiontype)
            down = compute_bsm_vanilla_price(forward - h, strike, ttm, vol, optiontype)
            assert abs(delta - (up - down) / (2.0 * h)) < 1e-4
