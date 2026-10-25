"""Generate independent literal fixtures using mpmath, not package pricing code."""

from pathlib import Path
import math
import mpmath as mp

mp.mp.dps = 110
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/vanilla_option_pricers/tests/_numerical_reference.py"
rows = []


def add(model, f, k, t, v, df=1.0, q="C"):
    f, k, t, v, df = map(float, (f, k, t, v, df))
    mf, mk, mt, mv, md = map(mp.mpf, (f, k, t, v, df))
    sign = 1 if q in ("C", "IC") else -1

    def price(vol):
        s = vol * mp.sqrt(mt)
        def cdf(x):
            return mp.erfc(-x / mp.sqrt(2)) / 2
        if model == "black":
            d1 = mp.log(mf / mk) / s + s / 2
            return md * sign * (mf * cdf(sign * d1) - mk * cdf(sign * (d1 - s)))
        d = (mf - mk) / s
        return md * (sign * (mf - mk) * cdf(sign * d) + s * mp.exp(-d * d / 2) / mp.sqrt(2 * mp.pi))

    p = float(price(mv))
    if p <= 0 or not math.isfinite(p):
        return
    intrinsic = md * max(sign * (mf - mk), 0)
    iv = None
    if mp.mpf(p) > intrinsic and (model != "black" or mp.mpf(p) < md * (mf if sign > 0 else mk)):
        # Solve for the *rounded input price*, not merely the generating sigma.
        scale = mp.mpf(p) - intrinsic
        try:
            iv = float(
                mp.findroot(
                    lambda vol: (price(vol) - mp.mpf(p)) / scale,
                    (mv, mv * mp.mpf("1.00001")),
                    tol=mp.mpf("1e-95"),
                )
            )
        except (ValueError, ZeroDivisionError):
            raise RuntimeError((model, f, k, t, v, q)) from None
    rows.append((model, f, k, t, v, df, q, p, iv))


for x in (0.0, -1e-12, -1e-6, -0.01, -0.1, -0.5, -1.0, -3.0, -10.0, -32.0, -200.0):
    for s in (0.02, 0.2, 0.5, 1.0, 3.0, 6.0, 12.0):
        add("black", math.exp(x / 2), math.exp(-x / 2), 1.0, s)
        add("black", math.exp(-x / 2), math.exp(x / 2), 1.0, s, q="P")
for h in (-4.0, -12.999999, -13.000001, -15.0, -25.0, -35.0):
    for t in (0.01, 0.21, 0.7):
        add("black", math.exp(h * t), math.exp(-h * t), 1.0, 2 * t)
for t in (1e-8, 1e-16, 1e-24, 1e-28, 1e-32):
    add("black", 100, 100, t, 0.2, 0.97)
for scale in (1e-200, 1e-8, 1e8, 1e200):
    add("black", scale, 1.1 * scale, 0.75, 0.4, 0.96)
for k in (math.nextafter(1.0, 2.0), 1.0 + 1e-12, 1.0 + 1e-8):
    for t in (1e-24, 1e-28, 1e-32):
        add("black", 1.0, k, t, 0.2)
for x in (-0.1, -1.0, -10.0):
    add("black", math.exp(x / 2), math.exp(-x / 2), 1.0, 16.0)
for q in ("C", "P", "IC", "IP"):
    add("black", 100, 110, 0.7, 0.35, 0.93, q)
    add("normal", -0.01, 0.005, 2.0, 0.012, 0.93, q)
for d in (0, 1, 4, 5, 7, 8, 9, 10, 15, 25, 35):
    add("normal", 100, 100 - d * 10, 1.0, 10, q="P")
    add("normal", 100 - d * 10, 100, 1.0, 10, q="C")
DEST.write_text(
    '''"""Independent mpmath 1.4.1 fixtures, generated at 110 decimal digits.

Each tuple is model, F, K, T, sigma, discount factor, option type, price,
and IV obtained by solving against that rounded price at high precision.
Computed from the defining Gaussian-integral formulas, not library kernels.
The generator is scripts/generate_numerical_reference.py (mpmath is optional
development tooling only). This file lets CI run with NumPy/Numba/pytest only.
"""

REFERENCE_CASES = [
'''
    + "".join("    " + repr(r) + ",\n" for r in rows)
    + "]\n",
    encoding="utf-8",
)
print("Saved", len(rows), "independent fixtures")
