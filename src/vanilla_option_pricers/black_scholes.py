"""
Black-Scholes-Merton (lognormal) model: prices, greeks, and implied volatilities.

Prices are quoted on the forward: forward = spot * exp((r - q) * ttm), discounted by
discfactor = exp(-r * ttm). The volatility argument `vol` is the lognormal volatility.

Option payoffs, with s_ttm = vol*sqrt(ttm), d1 = (log(F/K) + 0.5*s_ttm^2)/s_ttm,
d2 = d1 - s_ttm, and N the standard-normal cdf:
    call:  discfactor * (F * N(d1) - K * N(d2))
    put:   discfactor * (K * N(-d2) - F * N(-d1))
Put-call parity: C - P = discfactor * (F - K).

optiontype is one of 'C', 'P' (vanilla) or 'IC', 'IP' (inverse); inverse types share
the vanilla payoff branch here.

Numba loops over aligned numpy arrays are used in preference to `np.vectorize` for the
slice/chain helpers; per-scalar functions also expose `np.vectorize` wrappers.

Part of the vanilla-option-pricers package:
https://github.com/ArturSepp/VanillaOptionPricers
"""

# packages
import numpy as np
from numba import njit
from typing import Union
from numba.typed import List

# vanilla_option_pricers
from vanilla_option_pricers.utils import ncdf, npdf, ncdf_inv
from vanilla_option_pricers._black import (
    _implied_total_vol, _log_moneyness, _normalised_black,
)


@njit
def is_intrinsic(ttm: float, vol: float) -> bool:
    """
    test whether an option has degenerated to its intrinsic value.

    True when there is no diffusion left to price: ttm <= 0, vol <= 0, or vol is nan.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    vol : float
        volatility.

    Returns
    -------
    bool
        True if the option should be priced at intrinsic value.
    """
    if ttm <= 0.0 or vol <= 0.0 or np.isnan(vol):
        return True
    else:
        return False


"""
**************************
prices
****************************
"""


@njit
def compute_bsm_vanilla_price(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    Black-Scholes-Merton forward price of a vanilla option.

    With s_ttm = vol*sqrt(ttm), d1 = (log(F/K) + 0.5*s_ttm^2)/s_ttm, d2 = d1 - s_ttm:
        call: discfactor * (F * N(d1) - K * N(d2))
        put:  discfactor * (K * N(-d2) - F * N(-d1))
    Below the diffusion floor (`is_intrinsic`) the intrinsic payoff is returned.

    Parameters
    ----------
    forward : float
        forward price of the underlying, F = spot*exp((r-q)*ttm).
    strike : float
        option strike.
    ttm : float
        time to maturity, in years.
    vol : float
        lognormal volatility.
    optiontype : str, default 'C'
        one of 'C', 'IC' (call branch) or 'P', 'IP' (put branch).
    discfactor : float, default 1.0
        discount factor exp(-r*ttm) applied to the forward payoff.

    Returns
    -------
    float
        option price.

    Raises
    ------
    NotImplementedError
        if `optiontype` is not one of 'C', 'P', 'IC', 'IP'.
    """
    if optiontype == 'C' or optiontype == 'IC':
        intrinsic = max(forward - strike, 0.0)
    elif optiontype == 'P' or optiontype == 'IP':
        intrinsic = max(strike - forward, 0.0)
    else:
        raise NotImplementedError("optiontype")
    if is_intrinsic(ttm=ttm, vol=vol):
        return discfactor * intrinsic
    if forward <= 0 or strike <= 0:
        return np.nan
    x = -abs(_log_moneyness(forward, strike))
    scale = np.sqrt(forward) * np.sqrt(strike)
    time_value = scale * _normalised_black(x, vol * np.sqrt(ttm))
    return discfactor * (intrinsic + time_value)


compute_bsm_vanilla_price_vector = np.vectorize(compute_bsm_vanilla_price, doc='Vectorized `compute_bsm_vanilla_price`')


@njit
def compute_bsm_vanilla_slice_prices(ttm: float,
                                     forward: float,
                                     strikes: np.ndarray,
                                     vols: np.ndarray,
                                     optiontypes: np.ndarray,
                                     discfactor: float = 1.0
                                     ) -> np.ndarray:
    """
    bsm prices for an aligned slice of strikes, vols, and optiontypes at one expiry.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    vols : np.ndarray
        lognormal vols, aligned with `strikes`.
    optiontypes : np.ndarray
        per-strike option types, aligned with `strikes`.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    np.ndarray
        prices, aligned with `strikes`.
    """
    def f(strike: float, vol: float, optiontype: str) -> float:
        return compute_bsm_vanilla_price(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_prices = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        bsm_prices[idx] = f(strike, vol, optiontype)
    return bsm_prices


@njit
def compute_bsm_forward_grid_prices(ttm: float,
                                    forwards: np.ndarray,
                                    strike: float,
                                    vol: float,
                                    optiontype: str,
                                    discfactor: float = 1.0
                                    ) -> np.ndarray:
    """
    bsm prices for one option across an array of forwards (a forward grid).

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forwards : np.ndarray
        forward grid.
    strike : float
        option strike.
    vol : float
        lognormal volatility.
    optiontype : str
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    np.ndarray
        prices, aligned with `forwards`.
    """
    def f(forward: float) -> float:
        return compute_bsm_vanilla_price(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_prices = np.zeros_like(forwards)
    for idx, forward in enumerate(forwards):
        bsm_prices[idx] = f(forward)
    return bsm_prices


"""
**************************
deltas
**************************
"""


@njit
def compute_bsm_vanilla_delta(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float,
                              optiontype: str,
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm forward delta d(price)/d(forward).

    delta = discfactor * N(d1) for a call and -discfactor * N(-d1) for a put, with
    d1 = log(F/K)/s_ttm + 0.5*s_ttm; below the diffusion floor the digital-limit delta
    (0 or +/-1) is returned. Inverse types return 0 in the delta sign.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    vol : float
        lognormal volatility.
    optiontype : str
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float
        forward delta.

    Raises
    ------
    NotImplementedError
        for unsupported `optiontype` at the intrinsic boundary.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            bsm_deltas = 1.0 if forward >= strike else 0.0
        elif optiontype == 'P' or optiontype == 'IP':
            bsm_deltas = - 1.0 if forward <= strike else 0.0
        else:
            raise NotImplementedError(f"optiontype")
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_ttm + 0.5 * s_ttm
        if optiontype == 'C':
            d1_sign = 1.0
        elif optiontype == 'P':
            d1_sign = - 1.0
        else:
            d1_sign = 0.0
        bsm_deltas = discfactor * d1_sign * ncdf(d1_sign * d1)
    return bsm_deltas


compute_bsm_vanilla_delta_vector = np.vectorize(compute_bsm_vanilla_delta, doc='Vectorized `compute_bsm_vanilla_delta`')


@njit
def compute_bsm_vanilla_slice_deltas(ttm: float,
                                     forward: float,
                                     strikes: np.ndarray,
                                     vols: np.ndarray,
                                     optiontypes: np.ndarray
                                     ) -> Union[float, np.ndarray]:
    """
    bsm forward deltas for an aligned slice of strikes, vols, and optiontypes.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    vols : np.ndarray
        lognormal vols, aligned with `strikes`.
    optiontypes : np.ndarray
        per-strike option types, aligned with `strikes`.

    Returns
    -------
    float or np.ndarray
        forward deltas, aligned with `strikes`.
    """
    def f(strike: float, vol: float, optiontype: str) -> float:
        return compute_bsm_vanilla_delta(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype)

    bsm_deltas = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        bsm_deltas[idx] = f(strike, vol, optiontype)
    return bsm_deltas


@njit
def compute_bsm_vanilla_deltas_ttms(ttms: np.ndarray,
                                    forwards: np.ndarray,
                                    strikes_ttms: List[np.ndarray],
                                    vols_ttms: List[np.ndarray],
                                    optiontypes_ttms: List[np.ndarray],
                                    ) -> List[np.ndarray]:
    """
    bsm forward deltas for a chain: one strike/vol/optiontype slice per expiry.

    Parameters
    ----------
    ttms : np.ndarray
        expiries, in years.
    forwards : np.ndarray
        forward per expiry, aligned with `ttms`.
    strikes_ttms : List[np.ndarray]
        per-expiry strike slices.
    vols_ttms : List[np.ndarray]
        per-expiry vol slices.
    optiontypes_ttms : List[np.ndarray]
        per-expiry optiontype slices.

    Returns
    -------
    List[np.ndarray]
        per-expiry forward-delta slices.
    """
    deltas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        deltas_ttms.append(compute_bsm_vanilla_slice_deltas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return deltas_ttms


@njit
def compute_bsm_vanilla_grid_deltas(ttm: float,
                                    forwards: np.ndarray,
                                    strike: float,
                                    vol: float,
                                    optiontype: str,
                                    discfactor: float = 1.0
                                    ) -> np.ndarray:
    """
    bsm forward deltas for one option across an array of forwards (a forward grid).

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forwards : np.ndarray
        forward grid.
    strike : float
        option strike.
    vol : float
        lognormal volatility.
    optiontype : str
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    np.ndarray
        forward deltas, aligned with `forwards`.
    """
    def f(forward: float) -> float:
        return compute_bsm_vanilla_delta(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_deltas = np.zeros_like(forwards)
    for idx, forward in enumerate(forwards):
        bsm_deltas[idx] = f(forward)
    return bsm_deltas


def compute_bsm_strike_from_delta(ttm: float,
                                  forward: float,
                                  delta: float,
                                  vol: float
                                  ) -> Union[float, np.ndarray]:
    """
    invert a bsm forward delta to the strike that produces it, at fixed vol.

    strike = F * exp(-s_ttm * (N^{-1}(delta) - 0.5*s_ttm)) with s_ttm = vol*sqrt(ttm).
    The refined normal quantile retains near-double-precision accuracy in tail delta
    grids.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    delta : float
        target forward delta; positive for calls, negative for puts.
    vol : float
        lognormal volatility.

    Returns
    -------
    float or np.ndarray
        strike consistent with `delta`.
    """
    inv_delta = ncdf_inv(delta) if delta > 0.0 else -ncdf_inv(-delta)
    s_t = vol * np.sqrt(ttm)
    strike = forward*np.exp(-s_t*(inv_delta - 0.5 * s_t))
    return strike


"""
****************************
Vega
****************************
"""
@njit
def compute_bsm_vanilla_vega(ttm: float,
                             forward: float,
                             strike: float,
                             vol: float,
                             discfactor: float = 1.0
                             ) -> float:
    """
    bsm vega d(price)/d(vol).

    vega = discfactor * F * n(d1) * sqrt(ttm) with d1 = log(F/K)/s_ttm + 0.5*s_ttm; zero
    below the diffusion floor. The default discfactor=1.0 returns the forward
    (undiscounted) vega, unchanged from prior releases.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    vol : float
        lognormal volatility.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm); 1.0 gives the undiscounted (forward) vega.

    Returns
    -------
    float
        vega.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        vega = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        vega = discfactor * forward * npdf(d1) * np.sqrt(ttm)
    return vega


compute_bsm_vanilla_vega_vector = np.vectorize(compute_bsm_vanilla_vega, doc='Vectorized `compute_bsm_vanilla_vega`')


@njit
def compute_bsm_slice_vegas(ttm: float,
                            forward: float,
                            strikes: np.ndarray,
                            vols: np.ndarray,
                            optiontypes: np.ndarray = None
                            ) -> np.ndarray:
    """
    bsm vegas for an aligned slice of strikes and vols (undiscounted).

    vega = F * n(d1) * sqrt(ttm) with d1 = log(F/strikes)/sT + 0.5*sT.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    vols : np.ndarray
        lognormal vols, aligned with `strikes`.
    optiontypes : np.ndarray, optional
        unused; vega is optiontype-independent. Kept for signature symmetry.

    Returns
    -------
    np.ndarray
        vegas, aligned with `strikes`.
    """
    sT = vols * np.sqrt(ttm)
    d1 = np.log(forward / strikes) / sT + 0.5 * sT
    vegas = forward * npdf(d1) * np.sqrt(ttm)
    return vegas


@njit
def compute_bsm_vegas_ttms(ttms: np.ndarray,
                           forwards: np.ndarray,
                           strikes_ttms: List[np.ndarray],
                           vols_ttms: List[np.ndarray],
                           optiontypes_ttms: List[np.ndarray],
                           ) -> List[np.ndarray]:
    """
    bsm vegas for a chain: one strike/vol slice per expiry (undiscounted).

    Parameters
    ----------
    ttms : np.ndarray
        expiries, in years.
    forwards : np.ndarray
        forward per expiry, aligned with `ttms`.
    strikes_ttms : List[np.ndarray]
        per-expiry strike slices.
    vols_ttms : List[np.ndarray]
        per-expiry vol slices.
    optiontypes_ttms : List[np.ndarray]
        per-expiry optiontype slices (unused; kept for symmetry).

    Returns
    -------
    List[np.ndarray]
        per-expiry vega slices.
    """
    vegas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        vegas_ttms.append(compute_bsm_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


"""
****************************
Gamma
****************************
"""

@njit
def compute_bsm_vanilla_gamma(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float
                              ) -> float:
    """
    bsm forward gamma d^2(price)/d(forward)^2, undiscounted.

    gamma = n(d1) / (F * s_ttm) with d1 = log(F/K)/s_ttm + 0.5*s_ttm; zero below the
    diffusion floor.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    vol : float
        lognormal volatility.

    Returns
    -------
    float
        gamma.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        gamma = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        gamma = npdf(d1) / (forward*s_t)
    return gamma


compute_bsm_vanilla_gamma_vector = np.vectorize(compute_bsm_vanilla_gamma, doc='Vectorized `compute_bsm_vanilla_gamma`')


"""
****************************
Theta
****************************
"""


@njit
def compute_bsm_vanilla_theta(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float,
                              optiontype: str,
                              discfactor: float = 1.0,
                              discount_rate: float = 0.0
                              ) -> float:
    """
    bsm theta d(price)/d(ttm) with spot held fixed, sign-flipped to a decay.

    theta = -discfactor * F * n(d1) * vol / (2*sqrt(ttm))  -/+  r * discfactor * K * N(+/-d2),
    the volatility-decay term minus (call) or plus (put) the rate term; zero below the
    diffusion floor.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    vol : float
        lognormal volatility.
    optiontype : str
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).
    discount_rate : float, default 0.0
        continuously-compounded rate r used in the rate term.

    Returns
    -------
    float
        theta.

    Raises
    ------
    NotImplementedError
        if `optiontype` is not one of 'C', 'P', 'IC', 'IP'.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        theta = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        d2 = d1 - s_t
        if optiontype == 'C' or optiontype == 'IC':
            theta = -discfactor*forward * npdf(d1)*vol/(2.0*np.sqrt(ttm)) - discount_rate*discfactor*strike*ncdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            theta = -discfactor*forward * npdf(d1)*vol/(2.0*np.sqrt(ttm)) + discount_rate*discfactor*strike*ncdf(-d2)
        else:
            raise NotImplementedError(f"optiontype")
    return theta


compute_bsm_vanilla_theta_vector = np.vectorize(compute_bsm_vanilla_theta, doc='Vectorized `compute_bsm_vanilla_theta`')


"""
********************************
implied vols
*******************************
"""


@njit
def infer_bsm_ivols_from_model_slice_prices(ttm: float,
                                            forward: float,
                                            strikes: np.ndarray,
                                            optiontypes: np.ndarray,
                                            model_prices: np.ndarray,
                                            discfactor: float,
                                            vol_lower: float = 0.01,
                                            vol_upper: float = 5.0,
                                            max_iters: int = 100,
                                            is_bounds_to_nan: bool = True
                                            ) -> np.ndarray:
    """
    bsm implied vols for an aligned slice of model prices at one expiry.

    Nan or (near-)zero prices map to nan (or `vol_lower`); the rest are inverted with
    `infer_bsm_implied_vol`.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    optiontypes : np.ndarray
        per-strike option types, aligned with `strikes`.
    model_prices : np.ndarray
        prices to invert, aligned with `strikes`.
    discfactor : float
        discount factor exp(-r*ttm).
    vol_lower : float, default 0.01
        lower bound of the vol search bracket.
    vol_upper : float, default 5.0
        upper bound of the vol search bracket.
    max_iters : int, default 100
        maximum solver iterations.
    is_bounds_to_nan : bool, default True
        if True, return nan for prices that are not bracketed.

    Returns
    -------
    np.ndarray
        implied vols, aligned with `strikes`.
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        if np.isnan(model_price) or np.isclose(model_price, 0.0):
            model_vol_ttm[idx] = np.nan if is_bounds_to_nan else vol_lower
        else:
            model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                       given_price=model_price,
                                                       strike=strike,
                                                       optiontype=optiontype,
                                                       vol_lower=vol_lower,
                                                       vol_upper=vol_upper,
                                                       max_iters=max_iters)
    return model_vol_ttm


@njit
def infer_bsm_implied_vol(forward: float,
                          ttm: float,
                          strike: float,
                          given_price: float,
                          discfactor: float = 1.0,
                          optiontype: str = 'C',
                          tol: float = 1e-8,  # convergence tolerance on the implied vol
                          vol_lower: float = 0.01,
                          vol_upper: float = 5.0,
                          max_iters: int = 100,
                          is_bounds_to_nan: bool = True
                          ) -> float:
    """
    Black implied volatility from Jaeckel's transformed rational initial guess.

    The private normalized OTM kernel follows Let's Be Rational, sections 3-4,
    reference revision 1520 (2024). Usually two Householder corrections attain
    float64 accuracy; the ATM case uses sqrt(8/T)*erfinv(price/(discfactor*F)).
    Put-call parity removes intrinsic once and all subsequent evaluations use
    the OTM formula. The final volatility must lie in [vol_lower, vol_upper].
    Input rounding limits accuracy near intrinsic and the maximum price; lost
    time value cannot be recovered. See http://www.jaeckel.org/LetsBeRational.pdf.

    Parameters
    ----------
    forward : float
        forward price of the underlying.
    ttm : float
        time to maturity, in years.
    strike : float
        option strike.
    given_price : float
        observed option price to invert.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).
    optiontype : str, default 'C'
        'C'/'IC' or 'P'/'IP'; the existing payoff normalization is retained.
    tol : float, default 1e-8
        absolute convergence tolerance on the implied vol.
    vol_lower : float, default 0.01
        lower bound of the vol search bracket.
    vol_upper : float, default 5.0
        upper bound of the vol search bracket.
    max_iters : int, default 100
        maximum solver iterations.
    is_bounds_to_nan : bool, default True
        if True, return np.nan when the price is not bracketed (rather than a bound).

    Returns
    -------
    float
        implied vol, or np.nan if the price is outside the achievable range and
        `is_bounds_to_nan` is True.
    """
    if np.isnan(given_price) or given_price <= 0.0:
        return np.nan if is_bounds_to_nan else vol_lower
    if optiontype == 'C' or optiontype == 'IC':
        intrinsic, maximum = max(forward - strike, 0.0), forward
    elif optiontype == 'P' or optiontype == 'IP':
        intrinsic, maximum = max(strike - forward, 0.0), strike
    else:
        raise NotImplementedError("optiontype")
    if (not np.isfinite(forward) or not np.isfinite(strike)
            or not np.isfinite(ttm) or not np.isfinite(discfactor)
            or forward <= 0 or strike <= 0 or ttm <= 0 or discfactor <= 0
            or vol_lower < 0 or vol_upper < vol_lower):
        return np.nan
    # Subtract the observed intrinsic once, then evaluate only the OTM kernel.
    # No option-code reassignment: NumPy fixed-width strings remain Numba-compatible.
    target = given_price - discfactor * intrinsic
    if target <= 0:
        return np.nan if is_bounds_to_nan else vol_lower
    if given_price >= discfactor * maximum:
        return np.nan if is_bounds_to_nan else vol_upper
    scale = np.sqrt(forward) * np.sqrt(strike)
    beta = (target / discfactor) / scale
    x = -abs(_log_moneyness(forward, strike))
    sqrt_ttm = np.sqrt(ttm)
    if max_iters <= 0:
        if beta < _normalised_black(x, vol_lower * sqrt_ttm):
            return np.nan if is_bounds_to_nan else vol_lower
        if beta > _normalised_black(x, vol_upper * sqrt_ttm):
            return np.nan if is_bounds_to_nan else vol_upper
        return 0.5 * (vol_lower + vol_upper)
    if beta <= 0 or beta >= np.exp(x/2):
        return np.nan if is_bounds_to_nan else (vol_lower if beta <= 0 else vol_upper)
    # Form the upper-price gap before normalization to retain its small value.
    complement = ((discfactor * maximum - given_price) / discfactor) / scale
    total_vol = _implied_total_vol(beta, x, max_iters, tol * sqrt_ttm, complement)
    vol = total_vol / sqrt_ttm
    # Allow only floating-point endpoint noise; the public bracket is unchanged.
    lower_slack = 8 * np.finfo(np.float64).eps * max(abs(vol), abs(vol_lower))
    upper_slack = 8 * np.finfo(np.float64).eps * max(abs(vol), abs(vol_upper))
    if vol < vol_lower - lower_slack:
        return np.nan if is_bounds_to_nan else vol_lower
    if vol > vol_upper + upper_slack:
        return np.nan if is_bounds_to_nan else vol_upper
    return min(max(vol, vol_lower), vol_upper)


@njit
def infer_bsm_ivols_from_slice_prices(ttm: float,
                                      forward: float,
                                      discfactor: float,
                                      strikes: np.ndarray,
                                      optiontypes: np.ndarray,
                                      model_prices: np.ndarray,
                                      ) -> np.ndarray:
    """
    bsm implied vols for an aligned slice of prices at one expiry.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    discfactor : float
        discount factor exp(-r*ttm).
    strikes : np.ndarray
        strikes.
    optiontypes : np.ndarray
        per-strike option types, aligned with `strikes`.
    model_prices : np.ndarray
        prices to invert, aligned with `strikes`.

    Returns
    -------
    np.ndarray
        implied vols, aligned with `strikes`.
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
    return model_vol_ttm


@njit
def infer_bsm_ivols_from_model_chain_prices(ttms: np.ndarray,
                                            forwards: np.ndarray,
                                            discfactors: np.ndarray,
                                            strikes_ttms: List[np.ndarray],
                                            optiontypes_ttms: List[np.ndarray],
                                            model_prices_ttms: List[np.ndarray],
                                            ) -> List[np.ndarray]:
    """
    bsm implied vols for a whole chain: one price slice per expiry.

    Parameters
    ----------
    ttms : np.ndarray
        expiries, in years.
    forwards : np.ndarray
        forward per expiry, aligned with `ttms`.
    discfactors : np.ndarray
        discount factor per expiry, aligned with `ttms`.
    strikes_ttms : List[np.ndarray]
        per-expiry strike slices.
    optiontypes_ttms : List[np.ndarray]
        per-expiry optiontype slices.
    model_prices_ttms : List[np.ndarray]
        per-expiry price slices to invert.

    Returns
    -------
    List[np.ndarray]
        per-expiry implied-vol slices.
    """
    model_vol_ttms = List()
    for ttm, forward, discfactor, strikes, optiontypes, model_prices_ttm in zip(ttms, forwards, discfactors,
                                                                                strikes_ttms, optiontypes_ttms,
                                                                                model_prices_ttms):
        model_vol = np.zeros_like(strikes)
        for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices_ttm, optiontypes)):
            model_vol[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
        model_vol_ttms.append(model_vol)
    return model_vol_ttms


"""
********************************************
Digital prices
********************************************
"""

@njit
def compute_bsm_digital_price(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm cash-or-nothing digital price (unit payoff if in the money at expiry).

    price = discfactor * N(d2) for a digital call and discfactor * N(-d2) for a digital
    put, with d2 = (log(F/K) - 0.5*s_ttm^2)/s_ttm; the intrinsic 0/1 payoff below the
    diffusion floor.

    Parameters
    ----------
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    ttm : float
        time to maturity, in years.
    vol : float
        lognormal volatility.
    optiontype : str, default 'C'
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float
        digital price.

    Raises
    ------
    NotImplementedError
        if `optiontype` is not one of 'C', 'P', 'IC', 'IP'.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            price = 1.0 if forward >= strike else 0.0
        elif optiontype == 'P' or optiontype == 'IP':
            price = 1.0 if forward <= strike else 0.0
        else:
            raise NotImplementedError(f"optiontype")
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        if optiontype == 'C' or optiontype == 'IC':
            price = discfactor * ncdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            price = discfactor * ncdf(-d2)
        else:
            raise NotImplementedError(f"optiontype")

    return price


@njit
def compute_bsm_digital_delta(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm cash-or-nothing digital delta d(price)/d(forward).

    delta = +/- discfactor * n(d2) / (F * s_ttm) with d2 = (log(F/K) - 0.5*s_ttm^2)/s_ttm,
    positive for a digital call and negative for a digital put; zero below the diffusion
    floor.

    Parameters
    ----------
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    ttm : float
        time to maturity, in years.
    vol : float
        lognormal volatility.
    optiontype : str, default 'C'
        option type.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float
        digital delta.

    Raises
    ------
    NotImplementedError
        if `optiontype` is not one of 'C', 'P', 'IC', 'IP'.
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        delta = 0.0
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        pnorm = discfactor / (forward * s_ttm)
        if optiontype == 'C' or optiontype == 'IC':
            delta = pnorm * npdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            delta = - pnorm * npdf(d2)
        else:
            raise NotImplementedError(f"optiontype")

    return delta
