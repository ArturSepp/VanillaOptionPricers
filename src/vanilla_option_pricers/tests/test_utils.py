"""Independent accuracy and boundary checks for normal-distribution utilities."""

import math
from statistics import NormalDist

import numpy as np
from numba import njit

from vanilla_option_pricers import (
    compute_bsm_strike_from_delta,
    compute_normal_delta_to_strike,
)
from vanilla_option_pricers.utils import erfcc, inv_erf, ncdf, ncdf_inv

NORMAL = NormalDist()


def test_erfcc_and_ncdf_match_standard_library_reference():
    points = np.array(
        [
            -30.0,
            -10.0,
            -8.0,
            -6.0,
            -3.0,
            -1.0,
            np.nextafter(0.0, -1.0),
            0.0,
            np.nextafter(0.0, 1.0),
            1.0,
            3.0,
            6.0,
            8.0,
            10.0,
            30.0,
        ]
    )
    expected_erfc = np.array([math.erfc(float(x)) for x in points])
    expected_cdf = np.array([0.5 * math.erfc(-float(x) / math.sqrt(2.0)) for x in points])

    np.testing.assert_allclose(erfcc(points), expected_erfc, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(ncdf(points), expected_cdf, rtol=2e-15, atol=2e-15)
    assert erfcc(0.0) == 1.0
    assert ncdf(0.0) == 0.5
    assert np.isscalar(erfcc(0.25))


def test_distribution_ufuncs_work_inside_njit():
    @njit
    def evaluate(values, probabilities):
        return ncdf(values), ncdf_inv(probabilities)

    values = np.array([-3.0, 0.0, 3.0])
    probabilities = np.array([0.001, 0.5, 0.999])
    actual_cdf, actual_quantiles = evaluate(values, probabilities)
    expected_cdf = np.array([NORMAL.cdf(float(x)) for x in values])
    expected_quantiles = np.array([NORMAL.inv_cdf(float(p)) for p in probabilities])

    np.testing.assert_allclose(actual_cdf, expected_cdf, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(actual_quantiles, expected_quantiles, rtol=5e-14, atol=5e-13)


def test_ncdf_inv_matches_independent_normaldist_reference():
    probabilities = np.array(
        [
            1e-300,
            1e-100,
            1e-15,
            1e-12,
            1e-9,
            1e-6,
            1e-4,
            1e-3,
            1e-2,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            0.999,
            0.9999,
            1.0 - 1e-15,
        ]
    )
    expected = np.array([NORMAL.inv_cdf(float(p)) for p in probabilities])

    np.testing.assert_allclose(ncdf_inv(probabilities), expected, rtol=5e-14, atol=5e-13)


def test_inv_erf_matches_tail_stable_normaldist_reference():
    points = np.array(
        [
            -1.0 + 2e-15,
            -0.999999999999,
            -0.9999,
            -0.99,
            -0.9,
            -0.5,
            -1e-8,
            0.0,
            1e-8,
            0.5,
            0.9,
            0.99,
            0.9999,
            0.999999999999,
            1.0 - 2e-15,
        ]
    )

    def reference(x):
        if x < 0.0:
            return NORMAL.inv_cdf(0.5 * (1.0 + x)) / math.sqrt(2.0)
        return -NORMAL.inv_cdf(0.5 * (1.0 - x)) / math.sqrt(2.0)

    expected = np.array([reference(float(x)) for x in points])
    np.testing.assert_allclose(inv_erf(points), expected, rtol=5e-14, atol=5e-13)


def test_inverse_boundaries_and_invalid_inputs():
    quantiles = ncdf_inv(np.array([-0.1, 0.0, 0.5, 1.0, 1.1, np.nan]))
    assert math.isnan(quantiles[0])
    assert quantiles[1] == -math.inf
    assert quantiles[2] == 0.0
    assert quantiles[3] == math.inf
    assert math.isnan(quantiles[4])
    assert math.isnan(quantiles[5])

    inverse_erfs = inv_erf(np.array([-1.1, -1.0, 0.0, 1.0, 1.1, np.nan]))
    assert math.isnan(inverse_erfs[0])
    assert inverse_erfs[1] == -math.inf
    assert inverse_erfs[2] == 0.0
    assert inverse_erfs[3] == math.inf
    assert math.isnan(inverse_erfs[4])
    assert math.isnan(inverse_erfs[5])


def test_delta_to_strike_helpers_match_independent_quantile_reference():
    ttm = 0.5
    forward = 100.0
    delta = 0.001
    normal_quantile = NORMAL.inv_cdf(delta)

    bsm_vol = 0.2
    bsm_sdev = bsm_vol * math.sqrt(ttm)
    expected_bsm = forward * math.exp(-bsm_sdev * (normal_quantile - 0.5 * bsm_sdev))
    actual_bsm = compute_bsm_strike_from_delta(ttm, forward, delta, bsm_vol)
    assert math.isclose(actual_bsm, expected_bsm, rel_tol=1e-13, abs_tol=1e-12)

    normal_vol = 5.0
    expected_normal = forward - normal_vol * math.sqrt(ttm) * normal_quantile
    actual_normal = compute_normal_delta_to_strike(ttm, forward, delta, normal_vol)
    assert math.isclose(actual_normal, expected_normal, rel_tol=1e-13, abs_tol=1e-12)
