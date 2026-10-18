"""Figures for the Physics page: the room's k-space lattice and the
source-position selection rule for mode coupling.

    python3 scripts/physics_figures.py images

Writes images/k-lattice.png and images/mode-coupling.png. Room model and
sub positions are the ones in room_modes.py / room_layout.py.
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GRID, INK, SERIES, ACCENT, SOFT = ("#1a1a1a", "#404040", "#c9c9c9",
                                       "#3987e5", "#d95926", "#2e2e2e")
C = 343.0
LX, LY, LZ = 4.445, 3.658, 2.692
TONES = (39.9, 59.0, 120.0)
SUBS = {"S1": (0.35, 0.35), "S2": (4.09, 0.35),
        "S3": (0.35, 3.31), "S4": (4.09, 3.31)}
PILLOWS = [(2.80, 3.31), (3.40, 3.31)]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 11,
})

out = sys.argv[1] if len(sys.argv) > 1 else "images"

# --- k-space lattice (k_x, k_y) with resonance shells -------------------
kx0, ky0, kz0 = np.pi / LX, np.pi / LY, np.pi / LZ
fig, ax = plt.subplots(figsize=(7.2, 6.0))
for nx in range(5):
    for ny in range(5):
        kx, ky = nx * kx0, ny * ky0
        if nx == ny == 0:
            continue
        ax.plot(kx, ky, "o", color=SERIES, ms=6)
        ax.annotate(f"({nx},{ny},·)", (kx, ky), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK)
th = np.linspace(0, np.pi / 2, 200)
for f in TONES:
    k = 2 * np.pi * f / C
    ax.plot(k * np.cos(th), k * np.sin(th), color=ACCENT, lw=1.6)
    ax.text(0.25, np.sqrt(k**2 - 0.25**2) + 0.04,
            f"{f:g} Hz", color=ACCENT, fontsize=9)
    if k > kz0:
        kp = np.sqrt(k**2 - kz0**2)
        ax.plot(kp * np.cos(th), kp * np.sin(th), color=ACCENT, lw=1.2,
                ls="--")
        ax.text(0.25, np.sqrt(kp**2 - 0.25**2) + 0.04,
                f"{f:g} Hz, $n_z=1$", color=ACCENT, fontsize=9)
ax.set_xlim(0, 3.1)
ax.set_ylim(0, 3.1)
ax.set_aspect("equal")
ax.grid(True)
ax.set_xlabel(r"$k_x$ (rad/m)")
ax.set_ylabel(r"$k_y$ (rad/m)")
ax.set_title(r"Allowed $(k_x, k_y)$; solid: $n_z = 0$ shells, dashed: $n_z = 1$")
fig.tight_layout()
fig.savefig(f"{out}/k-lattice.png", dpi=150, facecolor=BG)

# --- selection rule: mode shapes on the floor with source positions ------
MODES = [((1, 0, 0), 38.6), ((1, 1, 0), 60.7), ((2, 2, 0), 121.4)]
x = np.linspace(0, LX, 220)
y = np.linspace(0, LY, 180)
X, Y = np.meshgrid(x, y)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (n, fn) in zip(axs, MODES):
    P = np.cos(n[0] * np.pi * X / LX) * np.cos(n[1] * np.pi * Y / LY)
    im = ax.pcolormesh(X, Y, P, cmap="RdBu_r", vmin=-1, vmax=1,
                       shading="auto")
    ax.contour(X, Y, P, levels=[0], colors=INK, linewidths=1.2,
               linestyles="--")
    for name, (sx, sy) in SUBS.items():
        ax.plot(sx, sy, "s", ms=9, mfc=ACCENT if name in ("S1", "S4") else
                "none", mec=ACCENT, mew=1.6)
        ax.annotate(name, (sx, sy), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    for (mx, my) in [(0, LY / 2), (LX, LY / 2), (LX / 2, 0), (LX / 2, LY),
                     (LX / 2, LY / 2)]:
        ax.plot(mx, my, "D", ms=8, mfc="none", mec="#e0e0e0", mew=1.6,
                clip_on=False)
    for (px, py) in PILLOWS:
        ax.plot(px, py, "*", ms=11, color="#7bc96f")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, LX + 0.1)
    ax.set_ylim(-0.1, LY + 0.1)
    ax.set_title(f"{n}  {fn:.1f} Hz")
    ax.set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
cb.set_label(r"$\psi_{\mathbf{n}}(x, y, 0)$")
fig.savefig(f"{out}/mode-coupling.png", dpi=150, facecolor=BG,
            bbox_inches="tight")
