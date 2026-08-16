"""
numerical primitives for the vanilla-option-pricers package.

Standard-normal cdf/pdf and their inverse, implemented as closed-form rational
approximations so they compile under numba `njit` and vectorise over numpy arrays
(scipy is not a dependency of this package).

Accuracy contract
-----------------
These are approximations, not machine-precision special functions. The stated error
bounds propagate into every price, greek, and implied vol in the package and cap the
attainable precision of the root-finders:
- `ncdf` / `npdf`: absolute error <= 1.2e-7 (Numerical Recipes `erfcc`).
- `ncdf_inv`:      relative error <= 1.3e-4 (Winitzki inverse-erf approximation).
All functions are compiled with `fastmath=True`.

Part of the vanilla-option-pricers package:
https://github.com/ArturSepp/VanillaOptionPricers
"""

# packages
import numpy as np
from numba import njit
from typing import Union


ONE_OVER_SQRT_TWO_PI = 0.3989422804014327  # = 1.0 / sqrt(2*pi)
ONE_OVER_SQRT_TWO = 0.7071067811865475  # = 1.0 / sqrt(2)
SQRT_TWO = 1.41421356237  # = sqrt(2)
TWO_OVER_PI = 0.63661977236  # = 2.0 / pi


@njit(cache=False, fastmath=True)
def erfcc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    complementary error function erfc(x) via a rational approximation.

    Implements the `erfcc` routine of Press et al., Numerical Recipes in C++ (2nd ed.,
    Cambridge University Press, p. 226). Maximum absolute error 1.2e-7 for any real x.

    Parameters
    ----------
    x : float or np.ndarray
        real argument(s).

    Returns
    -------
    float or np.ndarray
        approximation to erfc(x) = 1 - erf(x), same shape as `x`.
    """
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))))
    fcc = np.where(np.greater(x, 0.0), r, 2.0-r)
    return fcc


@njit(cache=False, fastmath=True)
def ncdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    standard-normal cumulative distribution function N(x).

    Computed as N(x) = 1 - 0.5 * erfc(x / sqrt(2)) using `erfcc`; absolute error
    inherits the 1.2e-7 bound of `erfcc`.

    Parameters
    ----------
    x : float or np.ndarray
        real argument(s).

    Returns
    -------
    float or np.ndarray
        N(x) in (0, 1), same shape as `x`.
    """
    return 1.0 - 0.5*erfcc(ONE_OVER_SQRT_TWO*x)


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


@njit(cache=False, fastmath=True)
def inv_erf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse error function erf^{-1}(x) via the Winitzki approximation.

    erf^{-1}(x) = sign(x) * sqrt( sqrt(B^2 - C) - B ) with B = 2/(pi*a) + ln(1-x^2)/2,
    C = ln(1-x^2)/a, a = 0.147 (see eq. 7 of S. Winitzki, "A handy approximation for the
    error function and its inverse"). Odd in x; largest absolute error about 1.3e-4 near
    |x| -> 1. Valid for x in (-1, 1).

    Parameters
    ----------
    x : float or np.ndarray
        argument(s) in (-1, 1).

    Returns
    -------
    float or np.ndarray
        approximation to erf^{-1}(x), same shape as `x`.
    """
    a = 0.147
    const = TWO_OVER_PI / a
    log_one_minus_x2 = np.log(1.0-np.square(x))
    invf = np.sqrt(- const - 0.5*log_one_minus_x2 + np.sqrt(np.square(const + 0.5*log_one_minus_x2) - log_one_minus_x2 / a))
    return np.sign(x) * invf


@njit(cache=False, fastmath=True)
def ncdf_inv(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse standard-normal cdf N^{-1}(x) (the quantile / probit function).

    N^{-1}(x) = sqrt(2) * erf^{-1}(2x - 1); relative error inherits the 1.3e-4 bound
    of `inv_erf`, which is coarse for delta-grid strike construction. Valid for
    x in (0, 1).

    Parameters
    ----------
    x : float or np.ndarray
        probability level(s) in (0, 1).

    Returns
    -------
    float or np.ndarray
        approximation to N^{-1}(x), same shape as `x`.
    """
    return SQRT_TWO*inv_erf(2.0*x-1.0)
