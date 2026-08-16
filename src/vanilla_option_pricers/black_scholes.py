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
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            price = np.maximum(forward - strike, 0.0)
        elif optiontype == 'P' or optiontype == 'IP':
            price = np.maximum(strike - forward, 0.0)
        else:
            raise NotImplementedError(f"optiontype")

    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        if optiontype == 'C' or optiontype == 'IC':
            price = discfactor * (forward * ncdf(d1) - strike * ncdf(d2))
        elif optiontype == 'P' or optiontype == 'IP':
            price = -discfactor * (forward * ncdf(-d1) - strike * ncdf(-d2))
        else:
            raise NotImplementedError(f"optiontype")
    return price


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
    Accuracy is limited by `ncdf_inv` (relative error ~1.3e-4), so this is coarse for
    fine delta grids.

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
    bsm implied vol from a price, by safeguarded Newton (Newton-Raphson with bisection
    fallback) on the bracket [vol_lower, vol_upper].

    Newton uses the analytic discounted vega d(price)/d(vol) = discfactor*F*n(d1)*sqrt(ttm);
    a step leaving the bracket, or one that fails to reduce the residual, falls back to
    bisection. For a vanilla in-the-money option the out-of-the-money counterpart is
    inverted via put-call parity C - P = discfactor*(F - K) for better conditioning; the
    implied vol is identical for the two by parity. Typical convergence is 5-8 pricer
    evaluations; terminal accuracy is bounded by the 1.2e-7 error of `ncdf`.

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
    f_lo = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=vol_lower, discfactor=discfactor, optiontype=solve_type) - target
    f_hi = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=vol_upper, discfactor=discfactor, optiontype=solve_type) - target
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
    f = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=rts, discfactor=discfactor, optiontype=solve_type) - target
    d1 = np.log(forward / strike) / (rts * sqrt_ttm) + 0.5 * rts * sqrt_ttm
    df = discfactor * forward * npdf(d1) * sqrt_ttm
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
        f = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=rts, discfactor=discfactor, optiontype=solve_type) - target
        d1 = np.log(forward / strike) / (rts * sqrt_ttm) + 0.5 * rts * sqrt_ttm
        df = discfactor * forward * npdf(d1) * sqrt_ttm
        if f < 0.0:
            xl = rts
        else:
            xh = rts
    return rts


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
