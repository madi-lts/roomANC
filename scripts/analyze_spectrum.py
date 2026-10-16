#!/usr/bin/env python3
"""Averaged spectrum + tonal peak census for a room-noise recording.

Usage: analyze_spectrum.py INPUT.wav OUTPUT.png [REPORT.txt]

Computes a Welch-averaged power spectral density of the (mono-mixed) input,
detects tonal peaks below FMAX_TONES by prominence over a median-filtered
baseline, refines each peak frequency by parabolic interpolation, and writes
a two-panel plot plus a text report.

Levels are dBFS (re full scale) — phone/camera recordings are uncalibrated,
so only relative levels and frequencies are meaningful.
"""
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FMAX_TONES = 300.0     # Hz, upper bound of the tone census
FMAX_ZOOM = 200.0      # Hz, zoom panel range
MIN_PROMINENCE_DB = 8.0

# gruvbox-light styling to match the project site
BG, GRID, INK, SERIES, ACCENT = "#fbf1c7", "#d5c4a1", "#3c3836", "#076678", "#af3a03"


def welch_db(x, fs, seg_seconds=4.0):
    """Welch PSD in dBFS/Hz. Segment length is shortened if x is too short."""
    nperseg = min(int(seg_seconds * fs), len(x))
    f, pxx = signal.welch(x, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=int(0.75 * nperseg), detrend="constant")
    return f, 10.0 * np.log10(np.maximum(pxx, 1e-30))


def tonal_peaks(f, db, fmax=FMAX_TONES, min_prom_db=MIN_PROMINENCE_DB):
    """Peaks whose level is >= min_prom_db above a median-filtered baseline.

    Returns (f_peak, level_db, prominence_db) rows, frequency-refined by
    parabolic interpolation on the dB spectrum.
    """
    sel = f <= fmax
    fs_, dbs = f[sel], db[sel]
    df = f[1] - f[0]
    # baseline over a ~20 Hz median window (odd length >= 3)
    k = max(3, int(20.0 / df) | 1)
    baseline = signal.medfilt(dbs, kernel_size=k)
    prom = dbs - baseline
    idx, _ = signal.find_peaks(prom, height=min_prom_db, distance=max(1, int(1.0 / df)))
    rows = []
    for i in idx:
        if 0 < i < len(dbs) - 1:
            a, b, c = dbs[i - 1], dbs[i], dbs[i + 1]
            delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        else:
            delta = 0.0
        rows.append((fs_[i] + delta * df, dbs[i], prom[i]))
    rows.sort(key=lambda r: -r[2])
    return rows


def main(in_wav, out_png, out_txt=None):
    fs, x = wavfile.read(in_wav)
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x -= x.mean()
    dur = len(x) / fs

    f, db = welch_db(x, fs)
    df = f[1] - f[0]
    peaks = tonal_peaks(f, db)

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 11,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)

    sel = (f >= 10) & (f <= 2000)
    ax1.semilogx(f[sel], db[sel], color=SERIES, lw=1.2)
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("PSD (dBFS/Hz)")
    ax1.set_title(f"Room noise spectrum — Welch average, Δf = {df:.2f} Hz, {dur:.1f} s record",
                  loc="left", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.6)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    selz = f <= FMAX_ZOOM
    ax2.plot(f[selz], db[selz], color=SERIES, lw=1.2)
    # markers for every detected peak; labels only for prominent ones,
    # staggered vertically when neighbors are too close to read
    LABEL_PROM_DB = 12.0
    last_fp, last_y = -1e9, -1e9
    for fp, lv, pr in sorted(peaks):
        if fp > FMAX_ZOOM:
            continue
        ax2.plot([fp], [lv], "o", ms=5, color=ACCENT)
        if pr >= LABEL_PROM_DB:
            y = lv + 3
            if fp - last_fp < 10.0 and abs(y - last_y) < 6.0:
                y = last_y + 6
            ax2.annotate(f"{fp:.1f} Hz", xy=(fp, lv), xytext=(fp, y),
                         ha="center", fontsize=10, color=ACCENT)
            last_fp, last_y = fp, y
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("PSD (dBFS/Hz)")
    ax2.set_title(f"Low-frequency detail (0–{FMAX_ZOOM:.0f} Hz) with detected tonal peaks",
                  loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.savefig(out_png, dpi=150)

    lines = [f"input: {in_wav}", f"fs = {fs} Hz, duration = {dur:.2f} s, resolution = {df:.3f} Hz",
             "", f"tonal peaks <= {FMAX_TONES:.0f} Hz "
                 f"(prominence >= {MIN_PROMINENCE_DB:.0f} dB over 20 Hz median baseline):",
             f"{'f (Hz)':>9}  {'level (dBFS/Hz)':>16}  {'prominence (dB)':>16}"]
    for fp, lv, pr in peaks:
        lines.append(f"{fp:9.2f}  {lv:16.1f}  {pr:16.1f}")
    if not peaks:
        lines.append("  (none)")
    report = "\n".join(lines)
    print(report)
    if out_txt:
        with open(out_txt, "w") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
