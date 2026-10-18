#!/usr/bin/env python3
"""Baseline measurement campaign analysis: calibrated spectra, level map, A/B.

Recordings are listed in a manifest CSV (columns: file, tag, x, y, z,
cal_file, ref_file, notes; paths relative to the manifest). Coordinates are
metres in the room frame of scripts/room_layout.py. A cal_file (miniDSP
UMIK-1 / REW text format) makes that recording's levels absolute dB SPL;
without one, levels stay relative dBFS.

Intermittent source (procedure, Phase 1 protocol): the HVAC and pumps cycle
on ~30 s - 1 min timescales, independently per machine, so every per-tone
level here is computed from source-ON frames only (2 s frames classified
on/off per tone; see roomanc_analysis.tone_gate). ref_file, when present,
names the fixed reference channel recorded simultaneously with that row:
the on/off state is then read from the reference (immune to the roving mic
sitting on a null), and the map/ab levels are reported as the ratio to the
reference's own on-level, cancelling source-level variation between points.
Points that caught < MIN_ON_INTERVALS on-intervals or < MIN_ON_SECONDS of
cumulative on-time for a cycling tone are flagged for a revisit.

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

from roomanc_analysis import (ACCENT, GRID, INK, MIN_ON_INTERVALS,
                              MIN_ON_SECONDS, ROOM_LX, ROOM_LY, SERIES,
                              apply_plot_style, drift_stats,
                              frame_spectrogram, load_wav, read_manifest,
                              tonal_peaks, tone_gate, track_tone, welch_db)

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


def ref_mode(rows):
    """True if every row has a ref_file, False if none does.

    Mixing referenced and unreferenced recordings in one comparison is
    refused: with a reference, levels are ratios to the fixed channel;
    without, they are absolute — not a common scale.
    """
    has = [r.ref_file is not None for r in rows]
    if all(has):
        return True
    if not any(has):
        return False
    raise SystemExit("error: manifest mixes rows with and without ref_file "
                     "in one comparison; give every row a reference "
                     "recording or none")


def gated_tone_levels(rec, tones, half_bw):
    """Per-tone source-ON level of one recording, reference-gated if possible.

    Returns {tone: dict} with keys
      level: ON-frame band level, calibrated if rec is (None: no ON frames)
      rel:   level minus the reference channel's own ON level (None: no ref)
      act:   the ToneGate describing source activity (the reference's when
             present — the fixed channel sees the on/off state even when the
             roving mic sits on a null; the recording's own otherwise)
    """
    fs, x = load_wav(rec.file)
    spec = frame_spectrogram(x, fs)
    spec_r = None
    if rec.ref_file:
        fs_r, xr = load_wav(rec.ref_file)
        spec_r = frame_spectrogram(xr, fs_r)
    out = {}
    for t in tones:
        g = tone_gate(x, fs, t, half_bw=half_bw, spec=spec)
        if spec_r is not None:
            gr = tone_gate(xr, fs_r, t, half_bw=half_bw, spec=spec_r)
            lv = g.level_over(gr.mask_at(g.t))
            ref_lv, act = gr.level_on_db, gr
        else:
            lv, ref_lv, act = g.level_on_db, None, g
        if lv is not None and rec.calibrated:
            lv = float(rec.cal.dbfs_to_spl(t, lv))
        rel = None if (lv is None or ref_lv is None) else lv - ref_lv
        out[t] = {"level": lv, "rel": rel, "act": act}
    return out


def incomplete(act):
    """True when a cycling tone was not captured well enough at this point.

    A steady gate (no off state seen) is not flagged: with a reference that
    means the source simply stayed on; without one it may also mean a null,
    which is exactly why the protocol wants ref_file set.
    """
    return not act.steady and not act.done()


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
        lines.append("")
        lines.append("source state per tone (2 s frames; levels above are "
                     "whole-recording averages, diluted by off-time):")
        spec = frame_spectrogram(x, fs)
        gates = {fp: tone_gate(x, fs, fp, spec=spec)
                 for fp, _, _ in peaks[:6]}
        for fp, lv, pr in peaks[:6]:
            g = gates[fp]
            on_lv = g.level_on_db
            if rec.calibrated and on_lv is not None:
                on_lv = float(rec.cal.dbfs_to_spl(fp, on_lv))
            off = g.level_off_db
            if rec.calibrated and off is not None:
                off = float(rec.cal.dbfs_to_spl(fp, off))
            if g.steady:
                lines.append(f"  {fp:7.2f} Hz: steady (no off state seen; "
                             f"frame-level spread {g.split_db:.1f} dB), "
                             f"on-level {on_lv:.1f} {units}")
            else:
                lines.append(
                    f"  {fp:7.2f} Hz: duty {100 * g.on_fraction:.0f}%, "
                    f"{g.n_on_intervals} on-interval(s), "
                    f"{g.on_seconds:.0f} s on; on-level {on_lv:.1f}, "
                    f"off {off:.1f} {units}")
        if dur >= DRIFT_MIN_SECONDS:
            lines.append("")
            lines.append("drift statistics (2 s frames, 50% overlap, "
                         "source-on frames only):")
            for fp, lv, pr in peaks[:4]:
                st = drift_stats(*track_tone(x, fs, fp, gate=gates.get(fp)))
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
    use_ref = ref_mode(rows)
    tones = pick_tones(rows, args.tones)
    os.makedirs(args.outdir, exist_ok=True)
    apply_plot_style(plt)

    table = []    # dict per (recording, tone)
    revisit = []  # (file, tone, reason)
    for rec in rows:
        levels = gated_tone_levels(rec, tones, args.half_bw)
        for t in tones:
            lv, rel, act = (levels[t][k] for k in ("level", "rel", "act"))
            name = os.path.basename(rec.file)
            if lv is None:
                revisit.append((name, t, "no source-on frames captured"))
                continue
            if incomplete(act):
                revisit.append(
                    (name, t,
                     f"only {act.n_on_intervals} on-interval(s), "
                     f"{act.on_seconds:.0f} s on-time "
                     f"(need >= {MIN_ON_INTERVALS}, {MIN_ON_SECONDS:.0f} s)"))
            table.append({"tone": t, "x": rec.x, "y": rec.y, "z": rec.z,
                          "level": lv, "rel": rel,
                          "on_fraction": act.on_fraction,
                          "on_s": act.on_seconds,
                          "n_on": act.n_on_intervals,
                          "steady": act.steady, "file": name})

    csv_path = os.path.join(args.outdir, "levels.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tone_hz", "x_m", "y_m", "z_m",
                    f"level_on_{units.replace(' ', '_')}", "rel_db",
                    "on_fraction", "on_s", "n_on_intervals", "steady",
                    "file"])
        for r in table:
            w.writerow([f"{r['tone']:.2f}", f"{r['x']:.3f}", f"{r['y']:.3f}",
                        f"{r['z']:.3f}", f"{r['level']:.2f}",
                        "" if r["rel"] is None else f"{r['rel']:.2f}",
                        f"{r['on_fraction']:.2f}", f"{r['on_s']:.0f}",
                        r["n_on"], int(r["steady"]), r["file"]])
    print(f"wrote {csv_path} ({len(table)} rows)")

    heights = sorted({r["z"] for r in table})
    for t in tones:
        for z in heights:
            pts = [(r["x"], r["y"], r["rel"] if use_ref else r["level"])
                   for r in table if r["tone"] == t and r["z"] == z
                   and (r["rel"] is not None if use_ref else True)]
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
            ax.set_title(f"{t:.1f} Hz source-on level map at z = {z:.2f} m "
                         f"({len(pts)} points)", loc="left", fontweight="bold")
            cb_label = (f"on-level re reference (dB), ±{args.half_bw:g} Hz"
                        if use_ref else
                        f"on-state band level ({units}), ±{args.half_bw:g} Hz")
            fig.colorbar(pc, ax=ax, label=cb_label)
            out = os.path.join(args.outdir, f"map-{t:.1f}Hz-z{z:.2f}m.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")

    if revisit:
        print(f"\nrevisit ({len(revisit)} point-tone pairs under-captured):")
        for name, t, why in revisit:
            print(f"  {name} @ {t:.1f} Hz: {why}")


def cmd_ab(args):
    rows = read_manifest(args.manifest)
    a = [r for r in rows if r.tag == args.tag_a]
    b = [r for r in rows if r.tag == args.tag_b]
    if not a or not b:
        raise SystemExit(f"error: need rows tagged '{args.tag_a}' and "
                         f"'{args.tag_b}'")
    units = level_units(a + b)
    use_ref = ref_mode(a + b)
    tones = pick_tones(rows, args.tones)
    os.makedirs(args.outdir, exist_ok=True)

    def mean_levels(group):
        """Power-mean source-ON level per tone across a tag's recordings.

        With references, each value is the ratio to the fixed channel, so
        A and B stay comparable even when the source ran louder or longer
        in one condition.
        """
        vals = {t: [] for t in tones}
        for rec in group:
            levels = gated_tone_levels(rec, tones, args.half_bw)
            for t in tones:
                v = levels[t]["rel"] if use_ref else levels[t]["level"]
                if v is None:
                    print(f"warning: {os.path.basename(rec.file)} caught no "
                          f"source-on frames at {t:.1f} Hz; skipped")
                    continue
                if incomplete(levels[t]["act"]):
                    print(f"warning: {os.path.basename(rec.file)} "
                          f"under-captured {t:.1f} Hz "
                          f"({levels[t]['act'].n_on_intervals} on-interval(s), "
                          f"{levels[t]['act'].on_seconds:.0f} s)")
                vals[t].append(v)
        return {t: 10.0 * np.log10(np.mean(10.0 ** (np.asarray(v) / 10.0)))
                for t, v in vals.items() if v}

    la, lb = mean_levels(a), mean_levels(b)
    units_note = ("dB re reference, source-on" if use_ref
                  else f"{units}, source-on")
    lines = [f"A = {args.tag_a} ({len(a)} rec), B = {args.tag_b} "
             f"({len(b)} rec), levels in {units_note}, band ±{args.half_bw:g} Hz",
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
