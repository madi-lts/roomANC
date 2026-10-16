#!/usr/bin/env python3
"""Render the room layout diagram (top view + elevation) for the Layout page.

Usage: room_layout.py OUTPUT.png
Room: 4.445 x 3.658 m (14'7" x 12'0"), height 2.692 m (8'10").
Positions match layout.qmd.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

BG, GRID, INK, SERIES, ACCENT, SOFT = ("#fbf1c7", "#d5c4a1", "#3c3836",
                                       "#076678", "#af3a03", "#ebdbb2")
LX, LY, LZ = 4.445, 3.658, 2.692
R_NF = 0.93  # near-field exclusion radius ~ lambda/2pi at 59 Hz

# subs: phase 1 = diagonal pair S1/S4; phase 2 adds S2/S3
SUBS = {"S1": (0.35, 0.35, True), "S2": (4.09, 0.35, False),
        "S3": (0.35, 3.31, False), "S4": (4.09, 3.31, True)}
# (x, y, height)
MICS = {"M1": (1.55, 3.05, 0.60), "M2": (2.15, 3.05, 0.60),
        "M3": (3.30, 2.45, 1.20), "M4": (1.00, 1.70, 1.60),
        "M5": (2.60, 0.30, 1.20), "M6": (3.00, 1.90, 1.90),
        "M7": (1.05, 2.50, 1.00), "M8": (1.30, 1.10, 1.45)}

fig, (ax, axe) = plt.subplots(1, 2, figsize=(15, 7.2),
                              gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor(BG)

# ---------------- top view ----------------
ax.set_facecolor(BG)
ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, ec=INK, lw=2))

bed = Rectangle((1.40, 1.66), 1.5, 2.0, fc=SOFT, ec=GRID, lw=1)
dresser = Rectangle((2.30, 0.02), 1.52, 0.51, fc=SOFT, ec=GRID, lw=1)
ax.add_patch(bed)
ax.add_patch(dresser)
ax.text(2.15, 2.25, "bed\n(head at top wall)", ha="center", va="center",
        color=INK, fontsize=10)
ax.text(3.35, 0.14, "double dresser (h = 0.9 m)", ha="center", va="center",
        color=INK, fontsize=8.5)

for name, (x, y, p1) in SUBS.items():
    ax.add_patch(Circle((x, y), R_NF, fill=False, ec=GRID, ls="--", lw=1.1))
    if p1:
        ax.add_patch(Rectangle((x - 0.17, y - 0.17), 0.34, 0.34,
                               fc=ACCENT, ec=INK, lw=1))
    else:
        ax.add_patch(Rectangle((x - 0.17, y - 0.17), 0.34, 0.34,
                               fill=False, ec=ACCENT, ls=(0, (3, 2)), lw=1.6))
    dx = 0.32 if x < LX / 2 else -0.32
    dy = 0.30 if y < LY / 2 else -0.30
    ax.text(x + dx, y + dy, name, ha="center", va="center",
            color=ACCENT, fontsize=12, fontweight="bold")

LABEL_OFFSETS = {"M1": (-0.12, 0.17, "right"), "M2": (0.12, 0.17, "left"),
                 "M3": (0.00, 0.19, "center"), "M4": (-0.14, 0.00, "right"),
                 "M5": (0.16, 0.12, "left"), "M6": (0.16, -0.05, "left"),
                 "M7": (-0.14, 0.00, "right"), "M8": (0.14, -0.08, "left")}
for name, (x, y, h) in MICS.items():
    ax.plot([x], [y], "o", ms=10, color=SERIES, mec=INK, mew=1)
    dx, dy, ha = LABEL_OFFSETS[name]
    ax.annotate(f"{name} ({h})", xy=(x, y), xytext=(x + dx, y + dy), ha=ha,
                color=SERIES, fontsize=10, fontweight="bold")

ax.text(0.02, -0.30, "filled squares: subwoofers, phase 1 (diagonal pair)",
        color=ACCENT, fontsize=9.5)
ax.text(0.02, -0.46, "dashed squares: phase-2 expansion to 4 subwoofers",
        color=ACCENT, fontsize=9.5)
ax.text(0.02, -0.62, "dots: error mics, (height m); dashed arcs: near-field "
        "exclusion r ≈ 0.93 m", color=INK, fontsize=9.5)
ax.set_xlim(-0.25, LX + 0.25)
ax.set_ylim(-0.78, LY + 0.25)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Top view — 4.445 × 3.658 m (14′7″ × 12′0″)",
             color=INK, fontsize=13, fontweight="bold", loc="left")

# ---------------- elevation (x-z) ----------------
axe.set_facecolor(BG)
axe.add_patch(Rectangle((0, 0), LX, LZ, fill=False, ec=INK, lw=2))

# vertical (0,0,1) mode pressure magnitude |cos(pi z / Lz)| as guide, drawn
# as a horizontal-amplitude profile along the left wall
z = np.linspace(0, LZ, 200)
amp = 0.5 * np.abs(np.cos(np.pi * z / LZ))
axe.plot(amp, z, color=GRID, lw=1.5)
axe.fill_betweenx(z, 0, amp, color=SOFT, alpha=0.7)
axe.text(0.62, LZ - 0.12, "|p(z)| of (0,0,1) mode, 63.7 Hz nominal\n(null at z = H/2 = 1.35 m)",
         color=INK, fontsize=9, va="top")

for name, (x, y, p1) in SUBS.items():
    if y < LY / 2:  # front corners drawn solid, rear dimmed via dashes
        axe.add_patch(Rectangle((x - 0.17, 0.02), 0.34, 0.34,
                                fc=ACCENT if p1 else BG, ec=ACCENT, lw=1.4,
                                ls="-" if p1 else (0, (3, 2))))
for name, (x, y, h) in MICS.items():
    axe.plot([x], [h], "o", ms=10, color=SERIES, mec=INK, mew=1)
    axe.annotate(name, xy=(x, h), xytext=(x + 0.07, h + 0.05),
                 color=SERIES, fontsize=10, fontweight="bold")
axe.axhline(LZ / 2, color=GRID, ls=":", lw=1.2)
axe.text(LX - 0.05, LZ / 2 + 0.04, "z = H/2 nodal plane", ha="right",
         color=INK, fontsize=9)

axe.set_xlim(-0.25, LX + 0.25)
axe.set_ylim(-0.35, LZ + 0.3)
axe.set_aspect("equal")
axe.axis("off")
axe.set_title("Elevation (x–z) — H = 2.692 m (8′10″)",
              color=INK, fontsize=13, fontweight="bold", loc="left")
axe.text(0.02, -0.28, "mic heights straddle the H/2 nodal plane of the "
         "vertical mode", color=INK, fontsize=9.5)

fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150, facecolor=BG, bbox_inches="tight")
