#!/usr/bin/env python3
"""Phase 1 level-vs-position measurement grid (procedure step 1.2).

Usage: measurement_grid.py OUTDIR
Writes to OUTDIR:
  measurement-grid.png      top-view diagram of the grid points
  manifest.grid.csv         manifest skeleton for analyze_campaign.py `map`
                            (fill in cal_file with the UMIK-1 serial file)

Room: 4.445 x 3.658 x 2.692 m, room frame of scripts/room_layout.py
(origin at the S1 corner).

Grid design. Regular grid with wall offset delta = 0.20 m (a round-base
stand can hold the mic there; walls are pressure antinodes of every
cos-mode, so near-wall rows pin the mode amplitudes):

  x_i = delta + i*dx, i = 0..8,  dx = (Lx - 2*delta)/8 = 0.506 m
  y_j = delta + j*dy, j = 0..6,  dy = (Ly - 2*delta)/6 = 0.543 m
  z_k in {0.65, 1.40} m          (pillow plane; upper plane near the
                                  (0,0,1) null z = Lz/2 = 1.346 m)

Odd point counts put the i=4 column exactly on the (1,0,0)/(1,1,0) null
plane x = Lx/2 and the j=3 row on the (0,1,0)/(1,1,0) null plane
y = Ly/2. The census tones sit at or below the ~120 Hz cluster, where the
highest-order mode components are nx <= 3, ny <= 2, nz <= 2; the spacing
gives ~3 points per half-wavelength along x (Lx/3 = 1.48 m) and y, well
inside the ~0.74 m Nyquist bound for nx = 3.

Placements: the full 9x7 = 63 points at z = 0.65 m, a 32-point
checkerboard ((i+j) even) at z = 1.40 m, plus the two pillow recording
points -- 97 single-mic placements, ~3h15 at ~2 min each, splittable by
height pass.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

BG, GRID, INK, SERIES, ACCENT, SOFT = ("#1a1a1a", "#404040", "#c9c9c9",
                                       "#3987e5", "#d95926", "#2e2e2e")
LX, LY, LZ = 4.445, 3.658, 2.692
DELTA = 0.20                   # wall offset (m)
NX, NY = 9, 7                  # points per axis (odd: mid planes included)
DX = (LX - 2 * DELTA) / (NX - 1)
DY = (LY - 2 * DELTA) / (NY - 1)
HEIGHTS = (0.65, 1.40)         # z_k (m): pillow plane, near-null plane

# Confirmed real layout (Madi, 2026-10-17: mirror of her index description
# in both axes): bed headboard at the y = Ly wall spanning x 2.48-3.74 m;
# dresser at the y = 0 wall spanning x 2.22-3.74 m; door on the y = Ly wall
# spanning x 0.56-1.37 m (roughly the i=1..i=2 columns). NOTE: the bed and
# pillow x-positions differ from layout.qmd / scripts/room_layout.py --
# reconcile the site layout pages separately.
BED = (2.475, 1.638, 1.26, 2.00)      # x, y, w, h (head at y = Ly)
DRESSER = (2.215, 0.02, 1.52, 0.51)

# pillow recording points at the actual bed head (y = Ly wall)
PILLOWS = {"left": (2.80, 3.31, 0.65), "right": (3.40, 3.31, 0.65)}

# door on the y = Ly wall: 0.81 m leaf, hinge at the right jamb,
# swinging inward.
DOOR_HINGE, DOOR_W = 1.365, 0.81


def grid_points():
    """Yield (i, j, k, x, y, z); k=1 is the (i+j)-even checkerboard."""
    for k, z in enumerate(HEIGHTS):
        for j in range(NY):
            for i in range(NX):
                if k == 1 and (i + j) % 2:
                    continue
                yield i, j, k, DELTA + i * DX, DELTA + j * DY, z


def write_manifest(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "tag", "x", "y", "z", "cal_file", "notes"])
        for name, (x, y, z) in PILLOWS.items():
            w.writerow([f"pillow-{name}.wav", "pillow",
                        f"{x:.3f}", f"{y:.3f}", f"{z:.2f}",
                        "umik1-SERIAL.txt", f"{name} pillow"])
        for i, j, k, x, y, z in grid_points():
            w.writerow([f"grid-i{i}j{j}-z{int(round(z * 100)):03d}.wav",
                        "grid", f"{x:.3f}", f"{y:.3f}", f"{z:.2f}",
                        "umik1-SERIAL.txt", ""])


def draw(path):
    fig, ax = plt.subplots(figsize=(10.5, 9.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, ec=INK, lw=2))

    ax.add_patch(Rectangle(BED[:2], BED[2], BED[3], fc=SOFT, ec=GRID, lw=1))
    ax.add_patch(Rectangle(DRESSER[:2], DRESSER[2], DRESSER[3], fc=SOFT,
                           ec=GRID, lw=1))
    ax.text(BED[0] + BED[2] / 2, 2.67, "bed\n(head at this wall)",
            ha="center", va="center", color=INK, fontsize=10)
    ax.text(DRESSER[0] + DRESSER[2] / 2, 0.27, "dresser", ha="center",
            va="center", color=INK, fontsize=9)

    # door on the y = Ly wall, spanning roughly the i=1..i=2 columns
    ax.plot([DOOR_HINGE - DOOR_W, DOOR_HINGE], [LY, LY], color=BG, lw=3,
            zorder=2)
    ax.plot([DOOR_HINGE, DOOR_HINGE], [LY - DOOR_W, LY], color=INK, lw=1.6)
    ax.add_patch(Arc((DOOR_HINGE, LY), 2 * DOOR_W, 2 * DOOR_W, angle=0,
                     theta1=180, theta2=270, ec=GRID, ls=":", lw=1.2))
    ax.text(DOOR_HINGE - DOOR_W / 2 - 0.12, LY - 0.56, "door", ha="center",
            va="center", color=INK, fontsize=9)

    # predicted null planes the grid lines land on
    ax.axvline(LX / 2, color=GRID, ls="--", lw=1.2)
    ax.axhline(LY / 2, color=GRID, ls="--", lw=1.2)
    ax.text(LX / 2 + 0.05, 0.90, "x = Lx/2 null\n(1,0,0),(1,1,0)",
            color=INK, fontsize=8.5)
    ax.text(2.05, LY / 2 + 0.16, "y = Ly/2 null\n(0,1,0),(1,1,0)",
            color=INK, fontsize=8.5)

    for i, j, k, x, y, _ in grid_points():
        if k == 0:
            ax.plot([x], [y], "o", ms=5.5, color=SERIES, mec=INK, mew=0.6,
                    zorder=3)
        else:
            ax.plot([x], [y], "o", ms=11, mfc="none", mec=ACCENT, mew=1.4,
                    zorder=3)
    for name, (x, y, _) in PILLOWS.items():
        ax.plot([x], [y], "x", ms=11, color=INK, mew=2.4, zorder=4)

    for i in range(NX):
        ax.text(DELTA + i * DX, LY + 0.16, f"i={i}", ha="center", va="top",
                color=SERIES, fontsize=9)
    for j in range(NY):
        ax.text(LX + 0.10, DELTA + j * DY, f"j={j}", ha="right", va="center",
                color=SERIES, fontsize=9)

    ax.text(LX - 0.02, LY + 0.52,
            "dots: 63 points at z = 0.65 m (pillow plane)   "
            "rings: +32 checkerboard points at z = 1.40 m   "
            "×: pillow recording points", color=INK, fontsize=9.5)
    ax.text(LX - 0.02, LY + 0.68,
            f"x_i = 0.20 + {DX:.3f} i,  y_j = 0.20 + {DY:.3f} j  (m);  "
            "97 placements total, ≈ 3¼ h at ~2 min each",
            color=INK, fontsize=9.5)
    ax.text(LX - 0.02, LY + 0.84,
            "view rotated 180°: origin (S1 corner) at top right, "
            "x increases to the left, y downward",
            color=INK, fontsize=9.5)

    # view rotated 180 deg (per Madi): invert both axes so the S1 origin
    # sits at the top right; coordinates and indices are unchanged.
    ax.set_xlim(LX + 0.42, -0.25)
    ax.set_ylim(LY + 1.02, -0.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Phase 1 level-vs-position grid — top view, "
                 "4.445 × 3.658 m", color=INK, fontsize=13,
                 fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")


if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    draw(os.path.join(outdir, "measurement-grid.png"))
    write_manifest(os.path.join(outdir, "manifest.grid.csv"))
    n = sum(1 for _ in grid_points()) + len(PILLOWS)
    print(f"wrote measurement-grid.png and manifest.grid.csv "
          f"({n} placements)")
