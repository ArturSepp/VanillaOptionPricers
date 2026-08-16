"""
Bachelier (normal) model: prices, greeks, and implied volatilities.

Prices are quoted on the forward: forward = spot * exp((r - q) * ttm), discounted by
discfactor = exp(-r * ttm). The volatility argument `vol` is a *relative* normal
volatility: the absolute normal standard deviation over the period is
    sdev = forward * vol * sqrt(ttm),
so `vol` is dimensionless and comparable in magnitude to a lognormal vol. Because sdev
carries the forward, this parametrisation assumes a positive forward.

Option payoffs, with d = (forward - strike) / sdev and N, n the standard-normal cdf/pdf:
    call:  discfactor * ((F - K) * N(d) + sdev * n(d))
    put:   discfactor * ((F - K) * (N(d) - 1) + sdev * n(d))
Put-call parity: C - P = discfactor * (F - K).

optiontype is one of 'C', 'P' (vanilla) or 'IC', 'IP' (inverse); inverse types share
the vanilla payoff branch here.

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


@njit(cache=False, fastmath=True)
def compute_normal_price(forward: float,
                         strike: float,
                         ttm: float,
                         vol: float,
                         discfactor: float = 1.0,
                         optiontype: str = 'C'
                         ) -> float:
    """
    Bachelier (normal) forward price of a vanilla option.

    With sdev = forward*vol*sqrt(ttm) and d = (forward - strike) / sdev:
        price = discfactor * ((F - K) * N(d) + sdev * n(d))            for a call
        price = discfactor * ((F - K) * (N(d) - 1) + sdev * n(d))      for a put

    Parameters
    ----------
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    ttm : float
        time to maturity, in years.
    vol : float
        relative normal volatility (absolute sdev = forward*vol*sqrt(ttm)).
    discfactor : float, default 1.0
        discount factor exp(-r*ttm) applied to the forward payoff.
    optiontype : str, default 'C'
        one of 'C', 'IC' (call branch) or 'P', 'IP' (put branch).

    Returns
    -------
    float
        option price.

    Raises
    ------
    NotImplementedError
        if `optiontype` is not one of 'C', 'P', 'IC', 'IP'.
    """
    sdev = forward*vol*np.sqrt(ttm)
    d = (forward - strike) / sdev
    if optiontype == 'C' or optiontype == 'IC':
        price = discfactor * ((forward-strike) * ncdf(d) + sdev * npdf(d))
    elif optiontype == 'P' or optiontype == 'IP':
        price = discfactor * ((forward - strike) * (ncdf(d)-1.0) + sdev * npdf(d))
    else:
        raise NotImplementedError(f"optiontype")

    return price


@njit(cache=False, fastmath=True)
def compute_normal_slice_prices(ttm: float,
                                forward: float,
                                strikes: np.ndarray,
                                vols: np.ndarray,
                                optiontypes: np.ndarray,
                                discfactor: float = 1.0
                                ) -> np.ndarray:
    """
    normal prices for an aligned slice of strikes, vols, and optiontypes at one expiry.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    vols : np.ndarray
        relative normal vols, aligned with `strikes`.
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
        return compute_normal_price(forward=forward,
                                    ttm=ttm,
                                    vol=vol,
                                    strike=strike,
                                    optiontype=optiontype,
                                    discfactor=discfactor)
    normal_prices = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        normal_prices[idx] = f(strike, vol, optiontype)
    return normal_prices


def compute_normal_delta_to_strike(ttm: float,
                                   forward: float,
                                   delta: float,
                                   vol: float
                                   ) -> Union[float, np.ndarray]:
    """
    invert a normal delta to the strike that produces it, at fixed vol.

    Using strike = forward - sdev * N^{-1}(delta) with sdev = forward*vol*sqrt(ttm);
    the put branch uses N^{-1}(1 + delta) = -N^{-1}(|delta|) by quantile symmetry.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    delta : float
        target normal delta; positive for calls, negative for puts.
    vol : float
        relative normal volatility.

    Returns
    -------
    float or np.ndarray
        strike consistent with `delta`.
    """
    inv_delta = ncdf_inv(delta) if delta > 0.0 else ncdf_inv(1.0+delta)
    sdev = forward * vol * np.sqrt(ttm)
    strike = forward - sdev*inv_delta
    return strike


@njit(cache=False, fastmath=True)
def compute_normal_delta_from_lognormal_vol(ttm: float,
                                            forward: float,
                                            strike: float,
                                            given_price: float,
                                            optiontype: str,
                                            discfactor: float = 1.0
                                            ) -> float:
    """
    normal delta implied by a given option price.

    Inverts the price to a relative normal vol, then evaluates the normal delta at that
    vol; degenerates to the intrinsic delta as ttm -> 0.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    given_price : float
        observed option price to invert.
    optiontype : str
        'C' or 'P'.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float
        normal delta.
    """
    if np.abs(ttm) < 1e-12:
        if optiontype == 'C' and forward > strike:
            delta = 1.0
        elif optiontype == 'P' and forward < strike:
            delta = -1.0
        else:
            delta = 0.0
    else:
        normal_vol = infer_normal_implied_vol(forward=forward, ttm=ttm, strike=strike,
                                              given_price=given_price, optiontype=optiontype, discfactor=discfactor)
        delta = compute_normal_delta(ttm=ttm, forward=forward, strike=strike, vol=normal_vol,
                                     optiontype=optiontype, discfactor=discfactor)
    return delta


@njit(cache=False, fastmath=True)
def compute_normal_delta(ttm: float,
                         forward: float,
                         strike: float,
                         vol: float,
                         optiontype: str,
                         discfactor: float = 1.0
                         ) -> float:
    """
    normal (Bachelier) delta d(price)/d(forward).

    With d = (forward - strike) / sdev, delta = discfactor * N(d) for a call and
    delta = -discfactor * N(-d) for a put.

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strike : float
        option strike.
    vol : float
        relative normal volatility.
    optiontype : str
        'C' or 'P'.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float
        normal delta; np.nan for unsupported optiontype.
    """
    sdev = forward * vol * np.sqrt(ttm)
    d = (forward - strike) / sdev
    if optiontype == 'C':
        normal_delta = discfactor * ncdf(d)
    elif optiontype == 'P':
        normal_delta = - discfactor * ncdf(-d)
    else:
        normal_delta = np.nan
    return normal_delta


@njit(cache=False, fastmath=True)
def compute_normal_slice_deltas(ttm: Union[float, np.ndarray],
                                forward: Union[float, np.ndarray],
                                strikes: Union[float, np.ndarray],
                                vols: Union[float, np.ndarray],
                                optiontypes: Union[np.ndarray],
                                discfactor: float = 1.0
                                ) -> Union[float, np.ndarray]:
    """
    normal deltas for an aligned slice of strikes, vols, and optiontypes.

    Vectorised form of `compute_normal_delta`: with d = (forward - strikes) / sdev and
    per-strike sign s = +1 for calls, -1 for puts, deltas = discfactor * s * N(s * d).

    Parameters
    ----------
    ttm : float or np.ndarray
        time to maturity, in years.
    forward : float or np.ndarray
        forward price of the underlying.
    strikes : float or np.ndarray
        strikes.
    vols : float or np.ndarray
        relative normal vols, aligned with `strikes`.
    optiontypes : np.ndarray
        per-strike option types ('C'/'P'), aligned with `strikes`.
    discfactor : float, default 1.0
        discount factor exp(-r*ttm).

    Returns
    -------
    float or np.ndarray
        normal deltas, aligned with `strikes`.
    """
    sdev = forward * vols * np.sqrt(ttm)
    d = (forward - strikes) / sdev
    d1_sign = np.where(np.array([op == 'C' for op in optiontypes]), 1.0, -1.0)
    normal_deltas = discfactor * d1_sign * ncdf(d1_sign * d)
    return normal_deltas


@njit(cache=False, fastmath=True)
def compute_normal_deltas_ttms(ttms: np.ndarray,
                               forwards: np.ndarray,
                               strikes_ttms: List[np.ndarray],
                               vols_ttms: List[np.ndarray],
                               optiontypes_ttms: List[np.ndarray],
                               ) -> List[np.ndarray]:
    """
    normal deltas for a chain: one strike/vol/optiontype slice per expiry.

    Parameters
    ----------
    ttms : np.ndarray
        expiries, in years.
    forwards : np.ndarray
        forward per expiry, aligned with `ttms`.
    strikes_ttms : List[np.ndarray]
        per-expiry strike slices.
    vols_ttms : List[np.ndarray]
        per-expiry relative-normal-vol slices.
    optiontypes_ttms : List[np.ndarray]
        per-expiry optiontype slices.

    Returns
    -------
    List[np.ndarray]
        per-expiry normal-delta slices.
    """
    deltas_ttms = List()
    for ttm, forward, vols, strikes, optiontypes in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        deltas_ttms.append(compute_normal_slice_deltas(ttm=ttm, forward=forward, strikes=strikes, vols=vols, optiontypes=optiontypes))
    return deltas_ttms


@njit(cache=False, fastmath=True)
def compute_normal_slice_vegas(ttm: float,
                               forward: float,
                               strikes: np.ndarray,
                               vols: np.ndarray,
                               optiontypes: np.ndarray = None
                               ) -> np.ndarray:
    """
    normal vegas for an aligned slice of strikes and vols.

    Derivative of the normal price with respect to the relative vol:
    vega = forward * n(d) * sqrt(ttm) with d = (forward - strikes) / sdev
    (undiscounted; multiply by discfactor for d(price)/d(vol)).

    Parameters
    ----------
    ttm : float
        time to maturity, in years.
    forward : float
        forward price of the underlying.
    strikes : np.ndarray
        strikes.
    vols : np.ndarray
        relative normal vols, aligned with `strikes`.
    optiontypes : np.ndarray, optional
        unused; vega is optiontype-independent. Kept for signature symmetry.

    Returns
    -------
    np.ndarray
        normal vegas, aligned with `strikes`.
    """
    sdev = forward*vols * np.sqrt(ttm)
    d = (forward - strikes) / sdev
    vegas = forward * npdf(d) * np.sqrt(ttm)
    return vegas


@njit(cache=False, fastmath=True)
def compute_normal_vegas_ttms(ttms: np.ndarray,
                              forwards: np.ndarray,
                              strikes_ttms: List[np.ndarray],
                              vols_ttms: List[np.ndarray],
                              optiontypes_ttms: List[np.ndarray],
                              ) -> List[np.ndarray]:
    """
    normal vegas for a chain: one strike/vol slice per expiry.

    Parameters
    ----------
    ttms : np.ndarray
        expiries, in years.
    forwards : np.ndarray
        forward per expiry, aligned with `ttms`.
    strikes_ttms : List[np.ndarray]
        per-expiry strike slices.
    vols_ttms : List[np.ndarray]
        per-expiry relative-normal-vol slices.
    optiontypes_ttms : List[np.ndarray]
        per-expiry optiontype slices (unused; kept for symmetry).

    Returns
    -------
    List[np.ndarray]
        per-expiry normal-vega slices.
    """
    vegas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        vegas_ttms.append(compute_normal_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


@njit(cache=False, fastmath=True)
def infer_normal_implied_vol(forward: float,
                             ttm: float,
                             strike: float,
                             given_price: float,
                             discfactor: float = 1.0,
                             optiontype: str = 'C',
                             tol: float = 1e-8,  # convergence tolerance on the implied vol
                             vol_lower: float = 0.01,  # relative-vol bracket lower bound
                             vol_upper: float = 10.0,  # relative-vol bracket upper bound
                             max_iters: int = 100,
                             is_bounds_to_nan: bool = True
                             ) -> float:
    """
    relative normal implied vol from a price, by safeguarded Newton (Newton-Raphson with
    bisection fallback) on the bracket [vol_lower, vol_upper].

    Newton uses the analytic normal vega d(price)/d(vol) = discfactor*forward*n(d)*sqrt(ttm);
    a step leaving the bracket, or one that fails to reduce the residual, falls back to
    bisection. For a vanilla in-the-money option the out-of-the-money counterpart is
    inverted via put-call parity C - P = discfactor*(F - K) for better conditioning; the
    implied vol is identical for the two by parity. Terminal accuracy is bounded by the
    1.2e-7 error of `ncdf`.

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
        'C'/'P' (parity switch applied) or 'IC'/'IP' (inverted directly).
    tol : float, default 1e-8
        absolute convergence tolerance on the implied vol.
    vol_lower : float, default 0.01
        lower bound of the relative-vol search bracket.
    vol_upper : float, default 10.0
        upper bound of the relative-vol search bracket.
    max_iters : int, default 100
        maximum solver iterations.
    is_bounds_to_nan : bool, default True
        if True, return np.nan when the price is not bracketed (rather than a bound).

    Returns
    -------
    float
        relative normal implied vol, or np.nan if the price is outside the achievable
        range and `is_bounds_to_nan` is True.
    """
    # non-positive prices carry no vol information
    if np.isnan(given_price) or given_price <= 0.0:
        return np.nan if is_bounds_to_nan else vol_lower

    # invert the OTM counterpart for vanillas via put-call parity for conditioning
    solve_type = optiontype
    target = given_price
    if optiontype == 'C' and strike < forward:
        target = given_price - discfactor * (forward - strike)
        solve_type = 'P'
    elif optiontype == 'P' and strike > forward:
        target = given_price - discfactor * (strike - forward)
        solve_type = 'C'
    if target <= 0.0:
        return np.nan if is_bounds_to_nan else vol_lower

    sqrt_ttm = np.sqrt(ttm)
    f_lo = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=vol_lower, discfactor=discfactor, optiontype=solve_type) - target
    f_hi = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=vol_upper, discfactor=discfactor, optiontype=solve_type) - target
    if f_lo * f_hi > 0.0:  # price not bracketed by [vol_lower, vol_upper]
        if is_bounds_to_nan:
            return np.nan
        return vol_lower if f_lo > 0.0 else vol_upper

    # orient bracket so the residual is negative at xl (price increases in vol)
    if f_lo < 0.0:
        xl, xh = vol_lower, vol_upper
    else:
        xl, xh = vol_upper, vol_lower
    rts = 0.5 * (vol_lower + vol_upper)
    dx_old = np.abs(vol_upper - vol_lower)
    dx = dx_old
    f = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=rts, discfactor=discfactor, optiontype=solve_type) - target
    sdev = forward * rts * sqrt_ttm
    df = discfactor * forward * npdf((forward - strike) / sdev) * sqrt_ttm
    for _ in range(max_iters):
        if ((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0 or np.abs(2.0 * f) > np.abs(dx_old * df):
            dx_old = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                break
        else:
            dx_old = dx
            dx = f / df
            temp = rts
            rts = rts - dx
            if temp == rts:
                break
        if np.abs(dx) < tol:
            break
        f = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=rts, discfactor=discfactor, optiontype=solve_type) - target
        sdev = forward * rts * sqrt_ttm
        df = discfactor * forward * npdf((forward - strike) / sdev) * sqrt_ttm
        if f < 0.0:
            xl = rts
        else:
            xh = rts
    return rts


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_model_slice_prices(ttm: float,
                                               forward: float,
                                               strikes: np.ndarray,
                                               optiontypes: np.ndarray,
                                               model_prices: np.ndarray,
                                               discfactor: float
                                               ) -> np.ndarray:
    """
    relative normal implied vols for an aligned slice of model prices at one expiry.

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

    Returns
    -------
    np.ndarray
        relative normal implied vols, aligned with `strikes` (np.nan where not invertible).
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                      given_price=model_price,
                                                      strike=strike,
                                                      optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_slice_prices(ttm: float,
                                         forward: float,
                                         discfactor: float,
                                         strikes: np.ndarray,
                                         optiontypes: np.ndarray,
                                         model_prices: np.ndarray
                                         ) -> List:
    """
    relative normal implied vols for an aligned slice of prices at one expiry.

    Alias of `infer_normal_ivols_from_model_slice_prices` with the discfactor-first
    argument order used by the chain-level helpers.

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
        relative normal implied vols, aligned with `strikes`.
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                      given_price=model_price,
                                                      strike=strike,
                                                      optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_chain_prices(ttms: np.ndarray,
                                         forwards: np.ndarray,
                                         discfactors: np.ndarray,
                                         strikes_ttms: List[np.ndarray],
                                         optiontypes_ttms: List[np.ndarray],
                                         model_prices_ttms: List[np.ndarray],
                                         ) -> List[np.ndarray]:
    """
    relative normal implied vols for a whole chain: one price slice per expiry.

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
        per-expiry relative-normal-implied-vol slices.
    """
    model_vol_ttms = List()
    for ttm, forward, discfactor, strikes, optiontypes, model_prices in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms, model_prices_ttms):
        model_vol_ttm = np.zeros_like(strikes)
        for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
            model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                          given_price=model_price,
                                                          strike=strike,
                                                          optiontype=optiontype)
        model_vol_ttms.append(model_vol_ttm)
    return model_vol_ttms
