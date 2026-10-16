#!/usr/bin/env python3
"""Baseline measurement campaign analysis: calibrated spectra, level map, A/B.

Recordings are listed in a manifest CSV (columns: file, tag, x, y, z,
cal_file, notes; paths relative to the manifest). Coordinates are metres in
the room frame of scripts/room_layout.py. A cal_file (miniDSP UMIK-1 / REW
text format) makes that recording's levels absolute dB SPL; without one,
levels stay relative dBFS.

Subcommands
-----------
spectra MANIFEST OUTDIR
    Per-recording Welch spectrum plot + tone census report; for recordings
    >= 60 s, drift statistics of each major tone (spec R4 input).
map MANIFEST OUTDIR [--tones F1,F2,...] [--half-bw HZ]
    Level-vs-position map: per-tone band level at each grid-tagged position,
    one heatmap per (tone, height) plus levels.csv for the modal-model
    overlay (workstream 2).
ab MANIFEST OUTDIR [--tag-a window-closed] [--tag-b window-open] [...]
    Per-tone level delta between two tags: the open-window structure-borne
    diagnostic.

Tones default to the census of the 'pillow'-tagged recording (else the
first row): peaks <= 150 Hz with >= 12 dB prominence, strongest 6.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from roomanc_analysis import (ACCENT, GRID, INK, ROOM_LX, ROOM_LY, SERIES,
                              apply_plot_style, band_level_db, drift_stats,
                              load_wav, read_manifest, tonal_peaks,
                              track_tone, welch_db)

FMAX_CONTROL = 150.0   # Hz, control band upper edge (spec R1)
AUTO_TONE_PROM_DB = 12.0
AUTO_TONE_MAX = 6
DRIFT_MIN_SECONDS = 60.0


def level_units(rows):
    """'dB SPL' if every row is calibrated, 'dBFS' if none is.

    Mixing calibrated and uncalibrated recordings in one comparison is
    refused: their levels are not on a common scale.
    """
    cal = [r.calibrated for r in rows]
    if all(cal):
        return "dB SPL"
    if not any(cal):
        return "dBFS"
    raise SystemExit("error: manifest mixes calibrated and uncalibrated "
                     "recordings in one comparison; give every row a "
                     "cal_file or none")


def analyzed_spectrum(rec):
    """Load one recording -> (fs, x, f, db) with calibration applied."""
    fs, x = load_wav(rec.file)
    f, db = welch_db(x, fs)
    if rec.calibrated:
        db = rec.cal.dbfs_to_spl(f, db)
    return fs, x, f, db


def pick_tones(rows, tones_arg):
    """Explicit --tones list, or census of the pillow/first recording."""
    if tones_arg:
        return [float(s) for s in tones_arg.split(",")]
    ref = next((r for r in rows if r.tag == "pillow"), rows[0])
    _, _, f, db = analyzed_spectrum(ref)
    peaks = [p for p in tonal_peaks(f, db, fmax=FMAX_CONTROL)
             if p[2] >= AUTO_TONE_PROM_DB][:AUTO_TONE_MAX]
    if not peaks:
        raise SystemExit(f"error: no tones >= {AUTO_TONE_PROM_DB:.0f} dB "
                         f"prominence found in {ref.file}; pass --tones")
    tones = sorted(p[0] for p in peaks)
    print(f"tones from {os.path.basename(ref.file)} census: "
          + ", ".join(f"{t:.1f}" for t in tones) + " Hz")
    return tones


def cmd_spectra(args):
    rows = read_manifest(args.manifest)
    os.makedirs(args.outdir, exist_ok=True)
    apply_plot_style(plt)
    for i, rec in enumerate(rows, start=1):
        fs, x, f, db = analyzed_spectrum(rec)
        dur = len(x) / fs
        units = "dB SPL" if rec.calibrated else "dBFS"
        peaks = tonal_peaks(f, db)
        stem = f"{i:02d}-{rec.tag}"
        pos = ("" if rec.x is None else
               f" at ({rec.x:.2f}, {rec.y:.2f}, {rec.z:.2f}) m")

        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        sel = f <= 2.0 * FMAX_CONTROL
        ax.plot(f[sel], db[sel], color=SERIES, lw=1.2)
        for fp, lv, pr in peaks:
            if fp <= 2.0 * FMAX_CONTROL:
                ax.plot([fp], [lv], "o", ms=5, color=ACCENT)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(f"PSD ({units}/Hz)")
        ax.set_title(f"{rec.tag}{pos} — {dur:.0f} s", loc="left",
                     fontweight="bold")
        ax.grid(True, alpha=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.savefig(os.path.join(args.outdir, stem + ".png"), dpi=150)
        plt.close(fig)

        lines = [f"input: {rec.file}", f"tag: {rec.tag}{pos}",
                 f"fs = {fs} Hz, duration = {dur:.2f} s, "
                 f"levels in {units}", "",
                 f"{'f (Hz)':>9}  {'level (' + units + '/Hz)':>18}  "
                 f"{'prominence (dB)':>16}"]
        for fp, lv, pr in peaks:
            lines.append(f"{fp:9.2f}  {lv:18.1f}  {pr:16.1f}")
        if dur >= DRIFT_MIN_SECONDS:
            lines.append("")
            lines.append("drift statistics (2 s frames, 50% overlap):")
            for fp, lv, pr in peaks[:4]:
                st = drift_stats(*track_tone(x, fs, fp))
                if st is None:
                    lines.append(f"  {fp:7.2f} Hz: track too short")
                    continue
                lines.append(
                    f"  {fp:7.2f} Hz: mean {st['f_mean']:.3f}, "
                    f"std {st['f_std']:.3f}, range [{st['f_min']:.3f}, "
                    f"{st['f_max']:.3f}] Hz, max rate "
                    f"{st['max_rate_hz_per_min']:.2f} Hz/min "
                    f"over {st['span_s']:.0f} s")
        report = "\n".join(lines)
        print(report, end="\n\n")
        with open(os.path.join(args.outdir, stem + ".txt"), "w") as fh:
            fh.write(report + "\n")


def cmd_map(args):
    rows = [r for r in read_manifest(args.manifest)
            if r.tag in ("grid", "pillow") and r.x is not None]
    if len(rows) < 3:
        raise SystemExit("error: need >= 3 positioned rows "
                         "(tag grid/pillow with x,y,z) for a map")
    units = level_units(rows)
    tones = pick_tones(rows, args.tones)
    os.makedirs(args.outdir, exist_ok=True)
    apply_plot_style(plt)

    table = []   # (tone, x, y, z, level, file)
    for rec in rows:
        _, _, f, db = analyzed_spectrum(rec)
        for t in tones:
            lv = band_level_db(f, db, t, half_bw=args.half_bw)
            if lv is not None:
                table.append((t, rec.x, rec.y, rec.z, lv,
                              os.path.basename(rec.file)))

    csv_path = os.path.join(args.outdir, "levels.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tone_hz", "x_m", "y_m", "z_m", f"level_{units.replace(' ', '_')}",
                    "file"])
        for row in table:
            w.writerow([f"{row[0]:.2f}", f"{row[1]:.3f}", f"{row[2]:.3f}",
                        f"{row[3]:.3f}", f"{row[4]:.2f}", row[5]])
    print(f"wrote {csv_path} ({len(table)} rows)")

    heights = sorted({row[3] for row in table})
    for t in tones:
        for z in heights:
            pts = [(x, y, lv) for (tt, x, y, zz, lv, _) in table
                   if tt == t and zz == z]
            if len(pts) < 3:
                continue
            xs, ys, lvs = map(np.asarray, zip(*pts))
            gx, gy = np.meshgrid(np.linspace(0, ROOM_LX, 120),
                                 np.linspace(0, ROOM_LY, 100))
            gz = griddata((xs, ys), lvs, (gx, gy), method="linear")

            fig, ax = plt.subplots(
                figsize=(8.5, 7), constrained_layout=True)
            pc = ax.pcolormesh(gx, gy, gz, cmap="magma", shading="auto")
            ax.scatter(xs, ys, s=28, color=INK, zorder=3)
            for xv, yv, lv in pts:
                ax.annotate(f"{lv:.1f}", xy=(xv, yv), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=INK)
            ax.add_patch(plt.Rectangle((0, 0), ROOM_LX, ROOM_LY, fill=False,
                                       ec=INK, lw=2))
            ax.set_xlim(-0.1, ROOM_LX + 0.1)
            ax.set_ylim(-0.1, ROOM_LY + 0.1)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{t:.1f} Hz level map at z = {z:.2f} m "
                         f"({len(pts)} points)", loc="left", fontweight="bold")
            fig.colorbar(pc, ax=ax, label=f"band level ({units}), "
                                          f"±{args.half_bw:g} Hz")
            out = os.path.join(args.outdir, f"map-{t:.1f}Hz-z{z:.2f}m.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")


def cmd_ab(args):
    rows = read_manifest(args.manifest)
    a = [r for r in rows if r.tag == args.tag_a]
    b = [r for r in rows if r.tag == args.tag_b]
    if not a or not b:
        raise SystemExit(f"error: need rows tagged '{args.tag_a}' and "
                         f"'{args.tag_b}'")
    units = level_units(a + b)
    tones = pick_tones(rows, args.tones)
    os.makedirs(args.outdir, exist_ok=True)

    def mean_levels(group):
        """Power-mean band level per tone across a tag's recordings."""
        out = {}
        for t in tones:
            vals = []
            for rec in group:
                _, _, f, db = analyzed_spectrum(rec)
                lv = band_level_db(f, db, t, half_bw=args.half_bw)
                if lv is not None:
                    vals.append(lv)
            if vals:
                out[t] = 10.0 * np.log10(np.mean(10.0 ** (np.asarray(vals) / 10.0)))
        return out

    la, lb = mean_levels(a), mean_levels(b)
    lines = [f"A = {args.tag_a} ({len(a)} rec), B = {args.tag_b} "
             f"({len(b)} rec), levels in {units}, band ±{args.half_bw:g} Hz",
             "",
             f"{'f (Hz)':>9}  {'A':>8}  {'B':>8}  {'B - A (dB)':>11}"]
    for t in tones:
        if t in la and t in lb:
            lines.append(f"{t:9.2f}  {la[t]:8.1f}  {lb[t]:8.1f}  "
                         f"{lb[t] - la[t]:11.1f}")
    lines += ["", "interpretation: a small |B - A| with the window open "
                  "points to a structure-borne path (see physics.qmd)."]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.outdir,
                           f"ab-{args.tag_a}-vs-{args.tag_b}.txt"), "w") as fh:
        fh.write(report + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("manifest")
        sp.add_argument("outdir")
        sp.add_argument("--tones", default=None,
                        help="comma-separated tone frequencies in Hz "
                             "(default: census of the pillow recording)")
        sp.add_argument("--half-bw", type=float, default=1.0,
                        help="half-bandwidth of the per-tone integration "
                             "band in Hz (default 1.0)")

    sp = sub.add_parser("spectra", help="per-recording spectrum + census + drift")
    sp.add_argument("manifest")
    sp.add_argument("outdir")
    sp.set_defaults(fn=cmd_spectra)

    sp = sub.add_parser("map", help="level-vs-position map")
    common(sp)
    sp.set_defaults(fn=cmd_map)

    sp = sub.add_parser("ab", help="per-tone delta between two tags")
    common(sp)
    sp.add_argument("--tag-a", default="window-closed")
    sp.add_argument("--tag-b", default="window-open")
    sp.set_defaults(fn=cmd_ab)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
