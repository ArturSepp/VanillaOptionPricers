"""Compare warmed Numba IV batches with a preserved pre-change module.

Run from the repository root:
    python examples/performance/bsm_iv_comparison.py --baseline PATH --output PATH

PATH for --baseline is an unchanged black_scholes.py saved before implementation,
or obtained from the desired Git revision. Both solvers receive identical arrays;
price construction, Python dispatch, and compilation are outside timed regions.
Timing is diagnostic, not a CI performance gate. NumPy and Numba only.
"""

from enum import Enum
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import numpy as np
import numba
from numba import njit

from vanilla_option_pricers import black_scholes as current


class Locals(Enum):
    COMPARE = 1


def _batch(solver):
    """Compile an identical array loop around either scalar solver."""

    @njit
    def evaluate(forwards, strikes, ttms, prices, codes, discounts):
        result = np.empty(prices.size)
        for i in range(prices.size):
            result[i] = solver(forwards[i], ttms[i], strikes[i], prices[i], discounts[i], codes[i])
        return result

    return evaluate


@njit
def _prices(forwards, strikes, ttms, vols, codes, discounts):
    """Prepare one common input-price array outside timed regions."""
    result = np.empty(forwards.size)
    for i in range(forwards.size):
        result[i] = current.compute_bsm_vanilla_price(
            forwards[i], strikes[i], ttms[i], vols[i], codes[i], discounts[i]
        )
    return result


def run_local(local: Locals):
    """Run paired benchmarks with reproducible inputs and record accuracy as well."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if local != Locals.COMPARE:
        raise ValueError(local)
    spec = importlib.util.spec_from_file_location("baseline_black_scholes", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    old, new = _batch(baseline.infer_bsm_implied_vol), _batch(current.infer_bsm_implied_vol)
    rng = np.random.default_rng(1520)
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "size": args.size,
        "repeats": args.repeats,
        "cases": {},
    }
    package_root = Path(current.__file__).parent
    output["current_source_hashes"] = {
        filename: hashlib.sha256((package_root / filename).read_bytes()).hexdigest()
        for filename in ("black_scholes.py", "_black.py", "_black_special.py", "utils.py")
    }
    for name in ("ordinary_otm", "atm", "tails", "mixed_itm_otm"):
        n = args.size
        f = np.full(n, 100.0)
        t = rng.uniform(0.03, 2.0, n)
        vols = rng.uniform(0.1, 0.8, n)
        x = rng.uniform(-0.15, 0.15, n)
        if name == "atm":
            x[:] = 0.0
        elif name == "tails":
            t[:] = 1.0
            vols[:] = 0.2
            x = -rng.uniform(8.0, 35.0, n) * vols
        k = f * np.exp(-x)
        codes = np.where(x <= 0, "C", "P")
        if name == "mixed_itm_otm":
            codes = np.where(np.arange(n) % 2 == 0, "C", "P")
        discounts = np.exp(-0.03 * t)
        prices = _prices(f, k, t, vols, codes, discounts)
        inputs = f, k, t, prices, codes, discounts
        old_result, new_result = old(*inputs), new(*inputs)  # JIT warm-up
        measurements = {"baseline": [], "new": []}
        for repeat in range(args.repeats):
            order = (("baseline", old), ("new", new))
            if repeat % 2:
                order = order[::-1]
            for label, function in order:
                start = perf_counter_ns()
                function(*inputs)
                measurements[label].append((perf_counter_ns() - start) / n)
        record = {}
        for label, result in [("baseline", old_result), ("new", new_result)]:
            finite = np.isfinite(result)
            error = np.abs(result[finite] - vols[finite])
            record[label] = {
                "median_ns_per_iv": statistics.median(measurements[label]),
                "samples_ns_per_iv": measurements[label],
                "nonfinite": int(np.sum(~finite)),
                "errors_above_1e-7": int(np.sum(error > 1e-7)),
                "max_absolute_iv_error": float(np.max(error)),
            }
        record["speedup"] = (
            record["baseline"]["median_ns_per_iv"] / record["new"]["median_ns_per_iv"]
        )
        output["cases"][name] = record
        print(name, json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_local(local=Locals.COMPARE)
