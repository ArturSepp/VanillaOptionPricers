"""Practical Bachelier workflows using annualised absolute normal volatility."""

import numpy as np

import vanilla_option_pricers as vop


def run_equity_forward_example() -> None:
    """Convert a relative normal quote, then price and invert one call."""
    forward = 100.0
    strike = 102.0
    ttm = 0.5
    discfactor = 0.98

    relative_normal_vol = 0.05
    absolute_normal_vol = relative_normal_vol * forward
    price = vop.compute_normal_price(
        forward, strike, ttm, absolute_normal_vol, discfactor, "C"
    )
    discounted_delta = vop.compute_normal_delta(
        ttm, forward, strike, absolute_normal_vol, "C", discfactor
    )
    implied_vol = vop.infer_normal_implied_vol(
        forward, ttm, strike, price, discfactor, "C"
    )

    target_forward_delta = 0.25
    delta_strike = vop.compute_normal_delta_to_strike(
        ttm, forward, target_forward_delta, absolute_normal_vol
    )
    recovered_forward_delta = vop.compute_normal_delta(
        ttm, forward, delta_strike, absolute_normal_vol, "C", 1.0
    )

    if abs(implied_vol - absolute_normal_vol) > 1e-7:
        raise RuntimeError("Equity-style normal-volatility inversion failed")
    if abs(recovered_forward_delta - target_forward_delta) > 1e-12:
        raise RuntimeError("Normal forward-delta strike conversion failed")

    print("equity_forward_example")
    print(
        f"relative_normal_vol={relative_normal_vol:.4%} "
        f"absolute_normal_vol={absolute_normal_vol:.6f}"
    )
    print(
        f"call_price={price:.8f} discounted_delta={discounted_delta:.8f} "
        f"implied_vol={implied_vol:.8f}"
    )
    print(
        f"target_forward_delta={target_forward_delta:.4f} "
        f"strike={delta_strike:.8f} recovered_delta={recovered_forward_delta:.8f}"
    )


def run_negative_rate_slice_example() -> None:
    """Mark and invert an OTM option slice when the forward rate is negative."""
    forward = -0.0025
    ttm = 2.0
    discfactor = 0.985
    strikes = np.array([-0.0150, -0.0100, -0.0050, 0.0000, 0.0050, 0.0100])
    absolute_normal_vols = np.array([0.0090, 0.0085, 0.0080, 0.0078, 0.0080, 0.0085])
    option_types = np.where(strikes < forward, "P", "C")

    prices = vop.compute_normal_slice_prices(
        ttm,
        forward,
        strikes,
        absolute_normal_vols,
        option_types,
        discfactor,
    )
    deltas = vop.compute_normal_slice_deltas(
        ttm,
        forward,
        strikes,
        absolute_normal_vols,
        option_types,
        discfactor,
    )
    discounted_vegas = discfactor * vop.compute_normal_slice_vegas(
        ttm, forward, strikes, absolute_normal_vols, option_types
    )
    implied_vols = vop.infer_normal_ivols_from_slice_prices(
        ttm,
        forward,
        discfactor,
        strikes,
        option_types,
        prices,
    )

    max_iv_error = float(np.max(np.abs(implied_vols - absolute_normal_vols)))
    parity_strike = 0.0
    call = vop.compute_normal_price(
        forward, parity_strike, ttm, 0.0080, discfactor, "C"
    )
    put = vop.compute_normal_price(
        forward, parity_strike, ttm, 0.0080, discfactor, "P"
    )
    parity_error = float(call - put - discfactor * (forward - parity_strike))

    if max_iv_error > 2e-7:
        raise RuntimeError(f"Rate-slice implied-volatility recovery failed: {max_iv_error}")
    if abs(parity_error) > 1e-12:
        raise RuntimeError(f"Bachelier put-call parity failed: {parity_error}")

    print("negative_rate_slice_example")
    print(f"forward={forward:.6f} ttm={ttm:.2f} discfactor={discfactor:.6f}")
    print("strikes=" + np.array2string(strikes, precision=6))
    print("option_types=" + str(option_types.tolist()))
    print("absolute_normal_vols=" + np.array2string(absolute_normal_vols, precision=6))
    print("prices=" + np.array2string(prices, precision=8))
    print("discounted_deltas=" + np.array2string(deltas, precision=8))
    print("discounted_vegas=" + np.array2string(discounted_vegas, precision=8))
    print(f"max_iv_error={max_iv_error:.3e} parity_error={parity_error:.3e}")


def main() -> None:
    """Run both normal-volatility examples."""
    run_equity_forward_example()
    print()
    run_negative_rate_slice_example()


if __name__ == "__main__":
    main()
