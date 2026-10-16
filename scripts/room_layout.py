#!/usr/bin/env python3
"""Render the nominal room layout diagram (top view) for the Layout page.

Usage: room_layout.py OUTPUT.png
Nominal room: 4.0 x 3.5 m (height 2.4 m). Positions match layout.qmd.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

BG, GRID, INK, SERIES, ACCENT, SOFT = ("#fbf1c7", "#d5c4a1", "#3c3836",
                                       "#076678", "#af3a03", "#ebdbb2")
LX, LY = 4.0, 3.5
R_NF = 0.93  # near-field exclusion radius ~ lambda/2pi at 59 Hz

SUBS = {"S1": (0.35, 0.35), "S2": (3.65, 0.35), "S3": (0.35, 3.15), "S4": (3.65, 3.15)}
MICS = {"M1": (1.40, 2.95, 0.6), "M2": (2.00, 2.95, 0.6),
        "M3": (2.80, 1.10, 1.2), "M4": (0.90, 1.60, 1.6)}

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, ec=INK, lw=2))

# furniture
bed = Rectangle((0.9, 2.0), 1.5, 1.5, fc=SOFT, ec=GRID, lw=1)
desk = Rectangle((2.35, 0.05), 1.0, 0.6, fc=SOFT, ec=GRID, lw=1)
ax.add_patch(bed)
ax.add_patch(desk)
ax.text(1.65, 2.35, "bed\n(head at top wall)", ha="center", va="center",
        color=INK, fontsize=10)
ax.text(2.85, 0.35, "desk", ha="center", va="center", color=INK, fontsize=10)

for name, (x, y) in SUBS.items():
    ax.add_patch(Circle((x, y), R_NF, fill=False, ec=GRID, ls="--", lw=1.2))
    ax.add_patch(Rectangle((x - 0.17, y - 0.17), 0.34, 0.34, fc=ACCENT, ec=INK, lw=1))
    dx = 0.30 if x < LX / 2 else -0.30
    dy = 0.30 if y < LY / 2 else -0.30
    ax.text(x + dx, y + dy, name, ha="center", va="center",
            color=ACCENT, fontsize=12, fontweight="bold")

for name, (x, y, h) in MICS.items():
    ax.plot([x], [y], "o", ms=11, color=SERIES, mec=INK, mew=1)
    left = name == "M1"
    ax.annotate(f"{name}  (h = {h} m)", xy=(x, y),
                xytext=(x - 0.14 if left else x + 0.14, y + 0.18),
                ha="right" if left else "left",
                color=SERIES, fontsize=11, fontweight="bold")

ax.text(0.03 * LX, -0.28, "squares: subwoofers (corner-loaded, floor level)",
        color=ACCENT, fontsize=10)
ax.text(0.03 * LX, -0.43, "dots: error microphones, h = height above floor",
        color=SERIES, fontsize=10)
ax.text(0.03 * LX, -0.58,
        "dashed: near-field exclusion, r = λ/2π ≈ 0.93 m at 59 Hz "
        "(no microphone inside)", color=INK, fontsize=10)

ax.set_xlim(-0.25, LX + 0.25)
ax.set_ylim(-0.75, LY + 0.25)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Nominal layout, top view — room 4.0 × 3.5 × 2.4 m",
             color=INK, fontsize=13, fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150, facecolor=BG, bbox_inches="tight")
