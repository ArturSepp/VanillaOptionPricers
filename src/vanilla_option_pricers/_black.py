"""Private numerical kernels for Black prices and implied volatility.

Selected routines adapted from Peter Jaeckel's Let's Be Rational.
See THIRD_PARTY_NOTICES.txt for attribution and licence terms.
"""

import math

import numpy as np
from numba import njit

# Keep fastmath disabled: tail accuracy depends on the evaluation order.

from vanilla_option_pricers._black_special import (
    SQRT_PI_OVER_TWO,
    TAU,
    _asymptotic,
    _erfcx,
    _lower_price_ratio,
    _one_minus_erfcx,
    _small_t,
    _upper_price_ratio,
)
from vanilla_option_pricers.utils import _ncdf_inv_scalar, inv_erf

_EPS = 2.220446049250313e-16
_MIN = 2.2250738585072014e-308
_SQRT_TWO = 1.4142135623730950488
_SQRT_TWO_PI = 2.5066282746310005024
_SQRT_THREE = 1.7320508075688772935
_R_MIN = -(1 - np.sqrt(_EPS))
_R_MAX = 2 / (_EPS * _EPS)


@njit
def _log_moneyness(forward: float, strike: float) -> float:
    """Log F/K, preserving nearby inputs and avoiding overflow of the ratio."""
    difference = forward - strike
    if abs(difference) < 0.5 * strike:
        return math.log1p(difference / strike)
    ratio = forward / strike
    if ratio > 0 and math.isfinite(ratio):
        return math.log(ratio)
    return math.log(forward) - math.log(strike)


@njit
def _general_black(x: float, s: float) -> float:
    """Normalized OTM Black price outside the two expansion regions; x <= 0."""
    h, t = x / s, s / 2
    q1, q2 = -(h + t) / _SQRT_TWO, -(h - t) / _SQRT_TWO
    if q1 < 0.46875:
        if q2 < 0.46875:
            twice = np.exp(x / 2) * math.erfc(q1) - np.exp(-x / 2) * math.erfc(q2)
        else:
            twice = np.exp(x / 2) * math.erfc(q1) - np.exp(-0.5 * (h * h + t * t)) * _erfcx(q2)
    else:
        twice = np.exp(-0.5 * (h * h + t * t)) * (_erfcx(q1) - _erfcx(q2))
    return max(0.5 * twice, 0.0)


@njit
def _normalised_black(x: float, s: float) -> float:
    """Normalized OTM Black price, equations (2.3)/(3.1), with 2024 regions."""
    if s <= 0:
        return 0.0
    if math.isinf(s):
        return np.exp(x / 2)
    if x == 0:
        return math.erf(s / np.sqrt(8.0))
    h, t = x / s, s / 2
    if h < -13 and t < TAU + 0.5 - 13 - h:
        return _asymptotic(h, t) * np.exp(-0.5 * (h * h + t * t)) / _SQRT_TWO_PI
    if t < TAU - h / 26:
        return _small_t(h, t) * np.exp(-0.5 * (h * h + t * t)) / _SQRT_TWO_PI
    return _general_black(x, s)


@njit
def _scaled_black(x: float, s: float) -> tuple[float, float]:
    """Compute b/bprime and log(bprime) jointly, avoiding tiny-price underflow."""
    h, t = x / s, s / 2
    log_vega = -0.5 * (h * h + t * t) - math.log(_SQRT_TWO_PI)
    if h < -13 and t < TAU + 0.5 - 13 - h:
        return _asymptotic(h, t), log_vega
    if t < TAU - h / 26:
        return _small_t(h, t), log_vega
    return _general_black(x, s) * np.exp(-log_vega), log_vega


@njit
def _rational_cubic(
    z: float, lo: float, hi: float, yl: float, yr: float, dl: float, dr: float, control: float
) -> float:
    """Delbourgo-Gregory rational cubic, equations (2.4)/(2.5)."""
    width = hi - lo
    if width == 0:
        return 0.5 * (yl + yr)
    t = (z - lo) / width
    omt = 1 - t
    if control >= _R_MAX:
        return yr * t + yl * omt
    return (
        yr * t * t * t
        + (control * yr - width * dr) * t * t * omt
        + (control * yl + width * dl) * t * omt * omt
        + yl * omt * omt * omt
    ) / (1 + (control - 3) * t * omt)


@njit
def _control(
    lo: float, hi: float, yl: float, yr: float, dl: float, dr: float, second: float, right:
    bool, preserve: bool
) -> float:
    """Fit curvature and impose monotonicity/convexity (Delbourgo-Gregory)."""
    slope = (yr - yl) / (hi - lo)
    numerator = 0.5 * (hi - lo) * second + dr - dl
    denominator = dr - slope if right else slope - dl
    if abs(denominator) < _MIN:
        r = _R_MAX if numerator > 0 else _R_MIN
    else:
        r = numerator / denominator
    monotone = dl * slope >= 0 and dr * slope >= 0
    convex = dl <= slope <= dr or dl >= slope >= dr
    rmin = _R_MIN
    if monotone:
        if abs(slope) >= _MIN:
            rmin = max(rmin, (dl + dr) / slope)
        elif preserve:
            rmin = _R_MAX
    if convex:
        if abs(slope - dl) >= _MIN and abs(dr - slope) >= _MIN:
            rmin = max(rmin, abs((dr - dl) / (dr - slope)), abs((dr - dl) / (slope - dl)))
        elif preserve:
            rmin = _R_MAX
    elif monotone and preserve:
        rmin = _R_MAX
    return max(r, rmin)


@njit
def _inverse_vega(x: float, s: float) -> float:
    """Reciprocal normalized vega for the rational interpolation endpoints."""
    h, t = x / s, s / 2
    return _SQRT_TWO_PI * np.exp(0.5 * (h * h + t * t))


@njit
def _rational_guess(beta: float, x: float) -> tuple[float, int]:
    """Initial total volatility from piecewise rational price interpolation."""
    bmax = np.exp(x / 2)
    root = np.sqrt(-x)
    sc = _SQRT_TWO * root
    ome = _one_minus_erfcx(root)
    bc = 0.5 * bmax * ome
    inv_vc = _SQRT_TWO_PI / bmax
    if beta < bc:
        sl = sc - SQRT_PI_OVER_TWO * ome
        bl = _lower_price_ratio(sc) * bmax
        if beta < bl:
            # Lower transformed map f and its first two beta derivatives.
            z = -x / (_SQRT_THREE * sl)
            y, s2 = z * z, sl * sl
            phi = np.exp(-y / 2) / _SQRT_TWO_PI
            cdf = 0.5 * math.erfc(z / _SQRT_TWO)
            f = 1.2091995761561452337 * (-x) * cdf * cdf * cdf
            fp = 2 * math.pi * y * cdf * cdf * np.exp(y + s2 / 8)
            fpp = (
                math.pi
                / 6
                * y
                / (s2 * sl)
                * cdf
                * (8 * _SQRT_THREE * sl * (-x) + (3 * s2 * (s2 - 8) - 8 * x * x) * cdf / phi)
                * np.exp(2 * y + s2 / 4)
            )
            r = _control(0.0, bl, 0.0, f, 1.0, fp, fpp, True, True)
            mapped = _rational_cubic(beta, 0.0, bl, 0.0, f, 1.0, fp, r)
            if not mapped > 0:
                t = beta / bl
                mapped = (f * t + bl * (1 - t)) * t
            # Separate cube roots avoid unnecessary underflow (2023 revision).
            p = 0.9386434874273835661 * mapped ** (1 / 3) / (-x) ** (1 / 3)
            s = abs(x / (_SQRT_THREE * _ncdf_inv_scalar(p)))
            return s, 0
        inv_vl = _inverse_vega(x, sl)
        r = _control(bl, bc, sl, sc, inv_vl, inv_vc, 0.0, True, False)
        return _rational_cubic(beta, bl, bc, sl, sc, inv_vl, inv_vc, r), 1
    su = sc + SQRT_PI_OVER_TWO * (2 - ome)
    bu = _upper_price_ratio(sc) * bmax
    if beta <= bu:
        inv_vu = _inverse_vega(x, su)
        r = _control(bc, bu, sc, su, inv_vc, inv_vu, 0.0, False, False)
        return _rational_cubic(beta, bc, bu, sc, su, inv_vc, inv_vu, r), 1
    w = (x / su) * (x / su)
    f = 0.5 * math.erfc(su / (2 * _SQRT_TWO))
    fp = -0.5 * np.exp(w / 2)
    fpp = SQRT_PI_OVER_TWO * np.exp(w + su * su / 8) * w / su
    mapped = -1.0
    if math.isfinite(fpp):
        r = _control(bu, bmax, f, 0.0, fp, -0.5, fpp, False, True)
        mapped = _rational_cubic(beta, bu, bmax, f, 0.0, fp, -0.5, r)
    if mapped <= 0:
        width = bmax - bu
        t = (beta - bu) / width
        mapped = (f * (1 - t) + 0.5 * width * t) * (1 - t)
    return -2 * _ncdf_inv_scalar(mapped), 2


@njit
def _householder(newton: float, h2: float, h3: float) -> float:
    """Third-order Householder correction (quartic local convergence)."""
    return newton * (1 + 0.5 * newton * h2) / (1 + newton * (h2 + newton * h3 / 6))


@njit
def _implied_total_vol(
    beta: float, x: float, max_iters: int, tolerance: float, complement: float
) -> float:
    """Invert normalized OTM price; two refinements normally attain float64 accuracy.

    Returns the last iterate at the caller's iteration limit, as the historical
    public solver does. max_iters and absolute total-vol tolerance remain active.
    The initial guess is independent of a caller's volatility bracket.
    """
    if x == 0:
        if complement < 0.5:
            return -2 * _ncdf_inv_scalar(complement / 2)
        return np.sqrt(8.0) * inv_erf(beta)
    s, branch = _rational_guess(beta, x)
    if not s > 0 or not math.isfinite(s):
        return math.nan
    log_beta = math.log(beta)
    for iteration in range(max_iters):
        h = x / s
        h2_black = h * h / s - s / 4
        h3_black = h2_black * h2_black - 3 * (h / s) * (h / s) - 0.25
        if branch == 0:
            bx, log_vega = _scaled_black(x, s)
            if not bx > 0:
                return math.nan
            log_b = math.log(bx) + log_vega
            bpob = 1 / bx
            inv_log = 1 / log_b
            newton = (log_beta - log_b) * log_b / log_beta * bx
            h2 = h2_black - bpob * (1 + 2 * inv_log)
            h3 = (
                h3_black
                + 2 * bpob * bpob * (1 + 3 * inv_log * (1 + inv_log))
                - 3 * h2_black * bpob * (1 + 2 * inv_log)
            )
        elif branch == 2:
            t = s / 2
            # Direct complement; do not subtract the calculated price from bmax.
            inv_gp = SQRT_PI_OVER_TWO * (_erfcx((t + h) / _SQRT_TWO) + _erfcx((t - h) / _SQRT_TWO))
            log_complement = math.log(inv_gp) - 0.5 * (h * h + t * t) - math.log(_SQRT_TWO_PI)
            gp = 1 / inv_gp
            newton = (log_complement - math.log(complement)) * inv_gp
            h2 = h2_black + gp
            h3 = h3_black + gp * (2 * gp + 3 * h2_black)
        else:
            newton = (beta - _normalised_black(x, s)) * _inverse_vega(x, s)
            h2, h3 = h2_black, h3_black
        ds = _householder(newton, h2, h3)
        if not math.isfinite(ds):
            return math.nan
        ds = max(-0.5 * s, ds)
        updated = s + ds
        if updated == s:
            return s
        s = updated
        # Do two corrections when available, then honor the requested tolerance.
        if iteration >= 1 and abs(ds) <= max(tolerance, _EPS * s):
            break
    return s
