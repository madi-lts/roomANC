#!/usr/bin/env python3
"""Averaged spectrum + tonal peak census for a room-noise recording.

Usage: analyze_spectrum.py INPUT.wav OUTPUT.png [REPORT.txt]

Computes a Welch-averaged power spectral density of the (mono-mixed) input,
detects tonal peaks below FMAX_TONES by prominence over a median-filtered
baseline, refines each peak frequency by parabolic interpolation, and writes
a two-panel plot plus a text report.

Levels are dBFS (re full scale) — phone/camera recordings are uncalibrated,
so only relative levels and frequencies are meaningful. For calibrated,
multi-recording campaigns (level maps, A/B tests, drift) see
analyze_campaign.py.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roomanc_analysis import (ACCENT, FMAX_TONES, MIN_PROMINENCE_DB, SERIES,
                              apply_plot_style, load_wav, tonal_peaks,
                              welch_db)

FMAX_ZOOM = 200.0      # Hz, zoom panel range


def main(in_wav, out_png, out_txt=None):
    fs, x = load_wav(in_wav)
    dur = len(x) / fs

    f, db = welch_db(x, fs)
    df = f[1] - f[0]
    peaks = tonal_peaks(f, db)

    apply_plot_style(plt)
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
