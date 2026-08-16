---
myst:
  html_meta:
    description: >-
      Normalize coin-denominated inverse-option quotes before using IC and IP branches in
      vanilla-option-pricers.
---

# Inverse-option normalization

Use this page when a market option is quoted and settled in units of the underlying coin.

```{warning}
`IC` and `IP` are branch labels, not a complete inverse-option conversion. They do not divide a
payoff or quote by spot or forward, change numeraire or probability measure, or compute the
premium-adjusted inverse hedge. Normalize the market quote before calling this package.
```

## Paper convention and required conversion

Lucic and Sepp, *Valuation and Hedging of Cryptocurrency Inverse Options*, Quantitative Finance
24(7), 851-869 (2024), defines the coin payoffs in Eq. (2):

$$
r_C(S_T)=\frac{(S_T-K)^+}{S_T}, \qquad
r_P(S_T)=\frac{(K-S_T)^+}{S_T}.
$$

Equation (8) values that payoff under the inverse measure. The change-of-numeraire result in
Eq. (10) is the operational conversion

$$
V_t^{currency}=S_t\,\widetilde V_t^{coin}.
$$

Equation (12) gives the premium-adjusted hedge in inverse-futures contract units:

$$
\widetilde\Delta_t=\Delta_t-\frac{V_t^{currency}}{S_t}.
$$

For the paper's futures-based Black convention, Eq. (13) is the regular currency value
$P(t,T)[F_tN(d_1)-KN(d_2)]$. The paper sets $S_t=P(t,T)F_t$, so the same coin quote can be
normalized in either consistent representation:

| Representation | Before pricing or IV | Convert package price back to coin |
|---|---|---|
| Forward numeraire, `discfactor=1.0` | `regular_value = forward * coin_quote` | `coin_quote = regular_value / forward` |
| Discounted currency value | `regular_value = spot * coin_quote` | `coin_quote = regular_value / spot` |

In the second row, `forward`, `spot`, and `discfactor` must follow one caller-owned carry model;
the paper's Black setup uses `spot = discfactor * forward`. Do not multiply a coin quote by spot
and then also treat it as the undiscounted forward-numeraire value.

## Executed forward-numeraire workflow

```python
from vanilla_option_pricers import (
    compute_bsm_vanilla_delta,
    compute_bsm_vanilla_price,
    compute_bsm_vanilla_theta,
    infer_bsm_implied_vol,
)

forward = 50_000.0
strike = 50_000.0
ttm = 7.0 / 365.0
vol = 0.60

regular_value = compute_bsm_vanilla_price(
    forward, strike, ttm, vol, "C", 1.0
)
ic_value = compute_bsm_vanilla_price(
    forward, strike, ttm, vol, "IC", 1.0
)
coin_quote = regular_value / forward
normalized_value = forward * coin_quote
implied = infer_bsm_implied_vol(
    forward, ttm, strike, normalized_value, 1.0, "IC"
)

black_delta = compute_bsm_vanilla_delta(
    ttm, forward, strike, vol, "C", 1.0
)
premium_adjusted_delta = black_delta - regular_value / forward
ic_delta = compute_bsm_vanilla_delta(
    ttm, forward, strike, vol, "IC", 1.0
)
theta_equal = compute_bsm_vanilla_theta(
    ttm, forward, strike, vol, "C"
) == compute_bsm_vanilla_theta(ttm, forward, strike, vol, "IC")

print(f"regular_value={regular_value:.8f} ic_value={ic_value:.8f}")
print(
    f"coin_quote={coin_quote:.10f} "
    f"normalized={normalized_value:.8f} iv={implied:.10f}"
)
print(
    f"black_delta={black_delta:.8f} "
    f"premium_adjusted={premium_adjusted_delta:.8f} ic_delta={ic_delta:.1f}"
)
print(f"price_equal={regular_value == ic_value} theta_equal={theta_equal}")
```

Expected output:

```text
regular_value=1656.95231139 ic_value=1656.95231139
coin_quote=0.0331390462 normalized=1656.95231139 iv=0.6000000000
black_delta=0.51656952 premium_adjusted=0.48343048 ic_delta=0.0
price_equal=True theta_equal=True
```

The equality is intentional: the package formula does not perform the coin conversion. The IV
round trip works because the coin quote was first multiplied by the forward. The inverse hedge
is computed explicitly from paper Eq. (12); `ic_delta=0.0` demonstrates why the package's `IC`
delta branch must not be used as that hedge.

## What `IC` and `IP` actually change

| Surface | Current behavior |
|---|---|
| BSM and normal prices | `IC` selects the ordinary call formula; `IP` selects the ordinary put formula. |
| BSM and normal IV | The normalized price is inverted directly, without the vanilla `C`/`P` parity-conditioning switch. |
| BSM scalar delta | `IC`/`IP` return zero when `ttm > 0` and `vol > 0`; the intrinsic branch returns the ordinary call/put limit. This is not Eq. (12). |
| Normal scalar delta | Returns `nan` for `IC`/`IP`; only `C` and `P` are implemented. |
| BSM theta and digitals | `IC`/`IP` reuse the ordinary call/put branches. |
| Gamma and vega helpers | They do not accept an option type, or ignore it; no inverse normalization is performed. |

The labels do **not** construct the terminal $1/S_T$ payoff, select the inverse measure, divide a
market quote by spot/forward, or convert the output units.

## Independent paper and model cross-check

The normalization above was checked against the
[published article](https://doi.org/10.1080/14697688.2024.2364804) and its
[SSRN record](https://ssrn.com/abstract=4606748), using payoff Eq. (2), inverse valuation Eq. (8),
value conversion Eq. (10), hedge Eq. (12), and Black value Eq. (13).

The independent `stochvolmodels` implementation keeps the missing responsibilities explicit:

- its [Monte Carlo payoff branch](https://github.com/ArturSepp/StochVolModels/blob/main/src/stochvolmodels/utils/mc_payoffs.py#L74-L86)
  divides `IC`/`IP` terminal payoffs by terminal spot;
- its [log-SV simulation](https://github.com/ArturSepp/StochVolModels/blob/main/src/stochvolmodels/pricers/logsv_pricer.py#L1032-L1045)
  changes the drift under the inverse measure; and
- its [Fourier payoff path](https://github.com/ArturSepp/StochVolModels/blob/main/src/stochvolmodels/utils/mgf_pricer.py#L193-L221)
  has separate spot- and inverse-measure transforms.

`vanilla-option-pricers` deliberately does none of those stochastic-model operations. Use
`stochvolmodels` when the task requires stochastic-volatility valuation rather than a normalized
Black/Bachelier kernel.

See the [pricing guide](pricing_and_greeks.md), [implied-volatility guide](implied_volatility.md),
[API inventory](api.md), and
[BSM source](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/black_scholes.py).
