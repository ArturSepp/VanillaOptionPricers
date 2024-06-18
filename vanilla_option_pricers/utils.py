"""
utility functions
"""
import numpy as np
from numba import njit
from typing import Union


ONE_OVER_SQRT_TWO_PI = 0.3989422804014327  # =1.0/sqrt(2.0*pi)
ONE_OVER_SQRT_TWO = 0.7071067811865475  # =1.0/sqrt(2.0)
SQRT_TWO = 1.41421356237  # = sqrt(2)
TWO_OVER_PI = 0.63661977236  # = 2.0/pi


@njit(cache=False, fastmath=True)
def erfcc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Complementary error function
    using algorithm from Press, William H. (2002), 2nd ed
    Numerical Recipes in C++: The Art of Scientific Computing. Cambridge University Press. p. 226.
    maximal error of 1.2×10−7 for any real argument
    """
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))))
    fcc = np.where(np.greater(x, 0.0), r, 2.0-r)
    return fcc


@njit(cache=False, fastmath=True)
def ncdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1.0 - 0.5*erfcc(ONE_OVER_SQRT_TWO*x)


@njit(cache=False, fastmath=True)
def npdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return ONE_OVER_SQRT_TWO_PI*np.exp(-0.5*np.square(x))


@njit(cache=False, fastmath=True)
def inv_erf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse of erf function
    x in (0, 1)
    See Eq 7 in A handy approximation for the error function and its inverse, Sergei Winitzki
    https://www.academia.edu/download/35916780/erf-approx.pdf
    largest relative error is about 1.3 · 10−4
    """
    a = 0.147
    const = TWO_OVER_PI / a
    log_one_minus_x2 = np.log(1.0-np.square(x))
    invf = np.sqrt(- const - 0.5*log_one_minus_x2 + np.sqrt(np.square(const+log_one_minus_x2) - log_one_minus_x2 / a))
    return invf


@njit(cache=False, fastmath=True)
def ncdf_inv(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse of normal cdf
    x in (0, 1)
    """
    return SQRT_TWO*inv_erf(2.0*x-1.0)

