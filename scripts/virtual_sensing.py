#!/usr/bin/env python3
"""Virtual (remote microphone) sensing analysis for the pillow positions.

Usage: virtual_sensing.py
Prints (stdout, markdown) the virtual-sensing error vs. displacement table
and the layout-rule / observability check of the relocated M1/M2 for the
Layout page.

Technique. At a tracked tone f, the pillow pressure is estimated from the
real error mics by one complex weight vector per tone,
  e_v(f) = w(f)^H e(f),
fitted in a calibration pass with a temporary mic at each pillow. This
script quantifies the honest failure mode: w is fitted on one modal
excitation and the modal mix then changes. Monte Carlo: draw random
complex modal amplitudes a_n (all rigid-wall modes to 225 Hz, damped
Lorentzian denominators as in room_modes.py), least-squares fit w from
the mics to the two pillow points on training draws, report the median
relative prediction error on fresh draws -- the ceiling on suppression
achievable at the pillow. Perturbed tests (tone moved 1 Hz; all
eigenfrequencies lowered 3% to mimic compliant walls) bound sensitivity
to drift.

Room and positions match scripts/room_layout.py and layout.qmd.
"""
import numpy as np
from itertools import product

rng = np.random.default_rng(7)
LX, LY, LZ, C, DF = 4.445, 3.658, 2.692, 343.0, 5.5


def freq(n):
    return (C / 2) * np.sqrt((n[0] / LX)**2 + (n[1] / LY)**2 + (n[2] / LZ)**2)


def psi(n, P):
    x, y, z = P
    return (np.cos(n[0] * np.pi * x / LX) * np.cos(n[1] * np.pi * y / LY)
            * np.cos(n[2] * np.pi * z / LZ))


NMAX = [int(np.ceil(2 * 225 * L / C)) for L in (LX, LY, LZ)]
MODES = [(n, freq(n)) for n in product(*(range(m + 1) for m in NMAX))
         if n != (0, 0, 0) and freq(n) <= 225]
FN = np.array([fn for _, fn in MODES])

PILLOWS = [(1.55, 3.30, 0.65), (2.15, 3.30, 0.65)]  # virtual points V1, V2
# fixed mics (M3-M6) and the relocated head-wall pair (see layout.qmd)
BASE = {"M3": (4.25, 1.45, 1.10), "M4": (0.20, 2.30, 1.75),
        "M5": (2.60, 0.30, 1.20), "M6": (2.95, 3.40, 2.10)}
M1_NEW, M2_NEW = (1.55, 3.40, 1.10), (2.60, 3.40, 0.65)


def field_matrix(points, f, fn=None):
    if fn is None:
        fn = FN
    G = np.array([[psi(n, P) for (n, _) in MODES] for P in points])
    return G / (fn**2 - f**2 + 1j * f * DF)


def vs_error(micpts, f, ntrain=400, ntest=400, fn_test=None, f_test=None):
    Gm, Gp = field_matrix(micpts, f), field_matrix(PILLOWS, f)
    A = rng.standard_normal((ntrain, len(MODES))) * (1 + 1j)
    W = np.linalg.lstsq(A @ Gm.T, A @ Gp.T, rcond=None)[0]
    Gm2 = field_matrix(micpts, f_test or f, fn_test)
    Gp2 = field_matrix(PILLOWS, f_test or f, fn_test)
    A2 = rng.standard_normal((ntest, len(MODES))) * (1 + 1j)
    err = np.abs(A2 @ Gm2.T @ W - A2 @ Gp2.T) / np.median(np.abs(A2 @ Gp2.T))
    return np.median(err)


def pair_at(d):
    """Pillow pair displaced by d: outboard along the head wall and up."""
    return [(max(0.25, 1.55 - d), 3.40, min(2.3, 0.65 + 0.7 * d)),
            (min(4.2, 2.15 + d), 3.40, min(2.3, 0.65 + 0.7 * d))]


print("| Pair displacement (m) | 59 Hz error | ceiling (dB) | 39.9 Hz "
      "error | +1 Hz drift | compliant walls (−3 %) |")
print("|---:|---:|---:|---:|---:|---:|")
for d in (0.0, 0.3, 0.6, 1.0, 1.5):
    mics = list(BASE.values()) + pair_at(d)
    e59 = vs_error(mics, 59.0)
    print(f"| {d:.1f} | {e59:.3f} | {20*np.log10(e59):.0f} | "
          f"{vs_error(mics, 39.9):.3f} | "
          f"{vs_error(mics, 59.0, f_test=60.0):.3f} | "
          f"{vs_error(mics, 59.0, fn_test=FN*0.97):.3f} |")

# the adopted relocation
mics = list(BASE.values()) + [M1_NEW, M2_NEW]
e59 = vs_error(mics, 59.0)
print(f"\n<!-- adopted M1={M1_NEW}, M2={M2_NEW}: 59 Hz error {e59:.3f} "
      f"({20*np.log10(e59):.0f} dB ceiling), "
      f"39.9 Hz {vs_error(mics, 39.9):.3f} -->")

# layout-rule check: horizontal distance to every (present or future) sub
SUBS = {"S1": (0.35, 0.35), "S2": (4.09, 0.35),
        "S3": (0.35, 3.31), "S4": (4.09, 3.31)}
ALL = dict(BASE, M1=M1_NEW, M2=M2_NEW)
dmin = min((np.hypot(mx - sx, my - sy), m, s)
           for m, (mx, my, mz) in ALL.items() for s, (sx, sy) in SUBS.items())
print(f"<!-- min horizontal mic-sub distance: {dmin[0]:.2f} m "
      f"({dmin[1]}-{dmin[2]}) -->")
