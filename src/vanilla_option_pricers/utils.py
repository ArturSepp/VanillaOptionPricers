"""
numerical primitives for the vanilla-option-pricers package.

Standard-normal cdf/pdf and their inverse, implemented with numba-compatible
standard-library functions and a refined rational quantile approximation. SciPy is
not a dependency of this package.

Accuracy contract
-----------------
`erfcc` and `ncdf` use the platform C-library `erfc`, avoiding cancellation in the
normal tails. `ncdf_inv` uses Acklam's piecewise rational approximation followed by
one lower-tail Halley refinement. Tests against Python's independent `NormalDist`
reference cover probabilities down to 1e-300 with near-double-precision agreement.

Part of the vanilla-option-pricers package:
https://github.com/ArturSepp/VanillaOptionPricers
"""

import math
from typing import Union

import numpy as np
from numba import float64, njit, vectorize


ONE_OVER_SQRT_TWO_PI = 0.3989422804014327  # = 1.0 / sqrt(2*pi)
ONE_OVER_SQRT_TWO = 0.7071067811865475  # = 1.0 / sqrt(2)


@vectorize([float64(float64)], nopython=True, cache=False)
def erfcc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    complementary error function erfc(x).

    Calls the platform C-library `erfc` directly from numba-compiled code. This removes
    the approximation error and the discontinuity at zero of the former Numerical
    Recipes implementation.

    Parameters
    ----------
    x : float or np.ndarray
        real argument(s).

    Returns
    -------
    float or np.ndarray
        erfc(x) = 1 - erf(x), same shape as `x`.
    """
    return math.erfc(x)


@vectorize([float64(float64)], nopython=True, cache=False)
def ncdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    standard-normal cumulative distribution function N(x).

    Computed as N(x) = 0.5 * erfc(-x / sqrt(2)). This form retains lower-tail
    probabilities that would be lost by subtracting a value close to one.

    Parameters
    ----------
    x : float or np.ndarray
        real argument(s).

    Returns
    -------
    float or np.ndarray
        N(x) in [0, 1], same shape as `x`.
    """
    return 0.5 * math.erfc(-ONE_OVER_SQRT_TWO * x)


@njit(cache=False, fastmath=True)
def npdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    standard-normal probability density function n(x).

    n(x) = exp(-x^2 / 2) / sqrt(2*pi), evaluated to machine precision.

    Parameters
    ----------
    x : float or np.ndarray
        real argument(s).

    Returns
    -------
    float or np.ndarray
        n(x), same shape as `x`.
    """
    return ONE_OVER_SQRT_TWO_PI*np.exp(-0.5*np.square(x))


@njit(float64(float64), cache=False)
def _ncdf_inv_scalar(probability: float) -> float:
    """Return one standard-normal quantile using a refined Acklam approximation."""
    if math.isnan(probability) or probability < 0.0 or probability > 1.0:
        return math.nan
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf

    upper_tail = probability > 0.5
    lower_probability = 1.0 - probability if upper_tail else probability

    if lower_probability < 0.02425:
        q = math.sqrt(-2.0 * math.log(lower_probability))
        quantile = (
            (((((-0.007784894002430293 * q - 0.3223964580411365) * q
                - 2.400758277161838) * q - 2.549732539343734) * q
              + 4.374664141464968) * q + 2.938163982698783)
            / ((((0.007784695709041462 * q + 0.3224671290700398) * q
                 + 2.445134137142996) * q + 3.754408661907416) * q + 1.0)
        )
    else:
        q = lower_probability - 0.5
        r = q * q
        quantile = (
            (((((-39.69683028665376 * r + 220.9460984245205) * r
                - 275.9285104469687) * r + 138.3577518672690) * r
              - 30.66479806614716) * r + 2.506628277459239) * q
            / (((((-54.47609879822406 * r + 161.5858368580409) * r
                  - 155.6989798598866) * r + 66.80131188771972) * r
                - 13.28068155288572) * r + 1.0)
        )

    # Refine against the lower tail. Evaluating the upper tail as 1 - N(x) would
    # discard the very probabilities for which the tail branch is needed.
    cdf_error = 0.5 * math.erfc(-ONE_OVER_SQRT_TWO * quantile) - lower_probability
    density = ONE_OVER_SQRT_TWO_PI * math.exp(-0.5 * quantile * quantile)
    correction = cdf_error / density
    quantile -= correction / (1.0 + 0.5 * quantile * correction)

    return -quantile if upper_tail else quantile


@vectorize([float64(float64)], nopython=True, cache=False)
def inv_erf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse error function erf^{-1}(x).

    Derived from the refined normal quantile with separate positive and negative tail
    mappings, so forming `1 + x` never destroys precision in the positive tail. A
    short central series preserves representable values close to zero.

    Parameters
    ----------
    x : float or np.ndarray
        argument(s) in [-1, 1]. Values outside this interval return NaN.

    Returns
    -------
    float or np.ndarray
        erf^{-1}(x), same shape as `x`; the endpoints return signed infinity.
    """
    if math.isnan(x) or x < -1.0 or x > 1.0:
        return math.nan
    if x == -1.0:
        return -math.inf
    if x == 1.0:
        return math.inf

    if abs(x) < 1e-4:
        x_squared = x * x
        return 0.886226925452758 * x * (
            1.0 + (math.pi / 12.0) * x_squared
            + (7.0 * math.pi * math.pi / 480.0) * x_squared * x_squared
        )

    if x < 0.0:
        return ONE_OVER_SQRT_TWO * _ncdf_inv_scalar(0.5 * (1.0 + x))
    return -ONE_OVER_SQRT_TWO * _ncdf_inv_scalar(0.5 * (1.0 - x))


@vectorize([float64(float64)], nopython=True, cache=False)
def ncdf_inv(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse standard-normal cdf N^{-1}(x) (the quantile / probit function).

    Uses Peter J. Acklam's piecewise rational approximation followed by one Halley
    refinement evaluated in the lower tail. This avoids both central cancellation and
    upper-tail probability loss.

    Parameters
    ----------
    x : float or np.ndarray
        probability level(s) in [0, 1]. Values outside this interval return NaN.

    Returns
    -------
    float or np.ndarray
        N^{-1}(x), same shape as `x`; 0 and 1 return signed infinity.
    """
    return _ncdf_inv_scalar(x)
