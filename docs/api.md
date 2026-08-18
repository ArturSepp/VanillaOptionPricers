---
myst:
  html_meta:
    description: Complete package-root API inventory for vanilla-option-pricers.
---

# Package-root API

The public import surface in `src/vanilla_option_pricers/__init__.py` contains 39 callable names.
The signatures below preserve the current source argument order, annotations, and defaults.
`List` means `numba.typed.List`; `np.ndarray` means a NumPy array. The five `*_vector` objects are
`numpy.vectorize` wrappers, whose runtime signature is `(*args, **kwargs)`.

## Look up a signature

```python
import inspect

from vanilla_option_pricers import compute_bsm_vanilla_price

print(inspect.signature(compute_bsm_vanilla_price))
```

Expected output on the supported Python 3.10-3.12 runtime:

```text
(forward: float, strike: float, ttm: float, vol: float, optiontype: str = 'C', discfactor: float = 1.0) -> float
```

For units, shapes, interpretation, and failure behavior, start with
[pricing and Greeks](pricing_and_greeks.md) or [implied volatility](implied_volatility.md).

## BSM scalar pricing, Greeks, and digitals

- `compute_bsm_vanilla_price(forward: float, strike: float, ttm: float, vol: float, optiontype: str = 'C', discfactor: float = 1.0) -> float` — vanilla price.
- `compute_bsm_vanilla_delta(ttm: float, forward: float, strike: float, vol: float, optiontype: str, discfactor: float = 1.0) -> float` — discounted forward delta.
- `compute_bsm_vanilla_vega(ttm: float, forward: float, strike: float, vol: float, discfactor: float = 1.0) -> float` — vega; default is undiscounted.
- `compute_bsm_vanilla_gamma(ttm: float, forward: float, strike: float, vol: float) -> float` — undiscounted forward gamma.
- `compute_bsm_vanilla_theta(ttm: float, forward: float, strike: float, vol: float, optiontype: str, discfactor: float = 1.0, discount_rate: float = 0.0) -> float` — calendar-decay theta with spot held fixed.
- `compute_bsm_digital_price(forward: float, strike: float, ttm: float, vol: float, optiontype: str = 'C', discfactor: float = 1.0) -> float` — unit cash-or-nothing price.
- `compute_bsm_digital_delta(forward: float, strike: float, ttm: float, vol: float, optiontype: str = 'C', discfactor: float = 1.0) -> float` — digital forward delta.
- `compute_bsm_strike_from_delta(ttm: float, forward: float, delta: float, vol: float) -> float | np.ndarray` — strike from target forward delta; accuracy is limited by the internal inverse-CDF approximation.

## BSM array and chain helpers

- `compute_bsm_vanilla_price_vector(*args, **kwargs)` — `numpy.vectorize` price convenience.
- `compute_bsm_vanilla_delta_vector(*args, **kwargs)` — `numpy.vectorize` delta convenience.
- `compute_bsm_vanilla_vega_vector(*args, **kwargs)` — `numpy.vectorize` vega convenience.
- `compute_bsm_vanilla_gamma_vector(*args, **kwargs)` — `numpy.vectorize` gamma convenience.
- `compute_bsm_vanilla_theta_vector(*args, **kwargs)` — `numpy.vectorize` theta convenience.
- `compute_bsm_vanilla_slice_prices(ttm: float, forward: float, strikes: np.ndarray, vols: np.ndarray, optiontypes: np.ndarray, discfactor: float = 1.0) -> np.ndarray` — aligned one-expiry prices.
- `compute_bsm_vanilla_slice_deltas(ttm: float, forward: float, strikes: np.ndarray, vols: np.ndarray, optiontypes: np.ndarray) -> float | np.ndarray` — aligned one-expiry forward deltas.
- `compute_bsm_slice_vegas(ttm: float, forward: float, strikes: np.ndarray, vols: np.ndarray, optiontypes: np.ndarray = None) -> np.ndarray` — aligned undiscounted vegas; option types are unused.
- `compute_bsm_forward_grid_prices(ttm: float, forwards: np.ndarray, strike: float, vol: float, optiontype: str, discfactor: float = 1.0) -> np.ndarray` — one option across a forward grid.
- `compute_bsm_vanilla_grid_deltas(ttm: float, forwards: np.ndarray, strike: float, vol: float, optiontype: str, discfactor: float = 1.0) -> np.ndarray` — its forward deltas.
- `compute_bsm_vanilla_deltas_ttms(ttms: np.ndarray, forwards: np.ndarray, strikes_ttms: List[np.ndarray], vols_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry delta slices.
- `compute_bsm_vegas_ttms(ttms: np.ndarray, forwards: np.ndarray, strikes_ttms: List[np.ndarray], vols_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry undiscounted vega slices.

## BSM implied volatility

- `infer_bsm_implied_vol(forward: float, ttm: float, strike: float, given_price: float, discfactor: float = 1.0, optiontype: str = 'C', tol: float = 1e-8, vol_lower: float = 0.01, vol_upper: float = 5.0, max_iters: int = 100, is_bounds_to_nan: bool = True) -> float` — safeguarded scalar inversion.
- `infer_bsm_ivols_from_model_slice_prices(ttm: float, forward: float, strikes: np.ndarray, optiontypes: np.ndarray, model_prices: np.ndarray, discfactor: float, vol_lower: float = 0.01, vol_upper: float = 5.0, max_iters: int = 100, is_bounds_to_nan: bool = True) -> np.ndarray` — configurable one-expiry inversion.
- `infer_bsm_ivols_from_slice_prices(ttm: float, forward: float, discfactor: float, strikes: np.ndarray, optiontypes: np.ndarray, model_prices: np.ndarray) -> np.ndarray` — discfactor-first slice convenience.
- `infer_bsm_ivols_from_model_chain_prices(ttms: np.ndarray, forwards: np.ndarray, discfactors: np.ndarray, strikes_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray], model_prices_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry chain inversion.

## Bachelier pricing and Greeks

- `compute_normal_price(forward: float, strike: float, ttm: float, vol: float, discfactor: float = 1.0, optiontype: str = 'C') -> float` — absolute-normal vanilla price.
- `compute_normal_delta(ttm: float, forward: float, strike: float, vol: float, optiontype: str, discfactor: float = 1.0) -> float` — discounted forward delta.
- `compute_normal_delta_from_lognormal_vol(ttm: float, forward: float, strike: float, given_price: float, optiontype: str, discfactor: float = 1.0) -> float` — normal delta after inverting the supplied price to absolute normal vol; despite the historical name, `given_price` is the input.
- `compute_normal_delta_to_strike(ttm: float, forward: float, delta: float, vol: float) -> float | np.ndarray` — strike from target normal delta.
- `compute_normal_slice_prices(ttm: float, forward: float, strikes: np.ndarray, vols: np.ndarray, optiontypes: np.ndarray, discfactor: float = 1.0) -> np.ndarray` — aligned one-expiry prices.
- `compute_normal_slice_deltas(ttm: float | np.ndarray, forward: float | np.ndarray, strikes: float | np.ndarray, vols: float | np.ndarray, optiontypes: np.ndarray, discfactor: float = 1.0) -> float | np.ndarray` — aligned normal deltas.
- `compute_normal_slice_vegas(ttm: float, forward: float, strikes: np.ndarray, vols: np.ndarray, optiontypes: np.ndarray = None) -> np.ndarray` — aligned undiscounted vegas.
- `compute_normal_deltas_ttms(ttms: np.ndarray, forwards: np.ndarray, strikes_ttms: List[np.ndarray], vols_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry delta slices.
- `compute_normal_vegas_ttms(ttms: np.ndarray, forwards: np.ndarray, strikes_ttms: List[np.ndarray], vols_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry undiscounted vega slices.

## Bachelier implied volatility

- `infer_normal_implied_vol(forward: float, ttm: float, strike: float, given_price: float, discfactor: float = 1.0, optiontype: str = 'C', tol: float = 1e-8, vol_lower: float = 1e-8, vol_upper: float = 1e4, max_iters: int = 100, is_bounds_to_nan: bool = True) -> float` — safeguarded scalar absolute-normal inversion.
- `infer_normal_ivols_from_model_slice_prices(ttm: float, forward: float, strikes: np.ndarray, optiontypes: np.ndarray, model_prices: np.ndarray, discfactor: float) -> np.ndarray` — one-expiry inversion using scalar defaults.
- `infer_normal_ivols_from_slice_prices(ttm: float, forward: float, discfactor: float, strikes: np.ndarray, optiontypes: np.ndarray, model_prices: np.ndarray) -> np.ndarray` — discfactor-first slice alias.
- `infer_normal_ivols_from_chain_prices(ttms: np.ndarray, forwards: np.ndarray, discfactors: np.ndarray, strikes_ttms: List[np.ndarray], optiontypes_ttms: List[np.ndarray], model_prices_ttms: List[np.ndarray]) -> List[np.ndarray]` — per-expiry chain inversion.

## Distribution helpers

- `ncdf(x: float | np.ndarray) -> float | np.ndarray` — standard-normal cumulative distribution approximation.
- `npdf(x: float | np.ndarray) -> float | np.ndarray` — standard-normal density.

The callable list is defined by the
[package root](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/__init__.py).
Unsupported option codes, unaligned bulk inputs, unbracketed prices, low-vega regimes, and Numba
container typing are described on the two task pages rather than repeated for every signature.
