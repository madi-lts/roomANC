"""Shared analysis routines for the roomANC measurement scripts.

Conventions
-----------
- Audio is loaded normalized to digital full scale, so uncalibrated levels
  are true dBFS (0 dBFS = full-scale sine has PSD-integrated power 1).
- A calibration file (miniDSP UMIK-1 / REW text format) turns dBFS into
  dB SPL: the header line ``"Sens Factor =-XX.XdB, SERNO: ..."`` gives the
  level in dBFS that a 94 dB SPL, 1 kHz tone produces, and the body lists
  ``frequency  gain_dB`` rows of correction to *add* to the measured
  spectrum (REW convention). Then

      SPL(f) = dBFS(f) + (94 - sens_dbfs) + corr(f).

- Room coordinates (x, y, z in metres) follow scripts/room_layout.py:
  origin at the S1 corner, room 4.445 x 3.658 x 2.692 m.
"""
import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.io import wavfile

FMAX_TONES = 300.0     # Hz, upper bound of the tone census
MIN_PROMINENCE_DB = 8.0

ROOM_LX, ROOM_LY, ROOM_LZ = 4.445, 3.658, 2.692

# dark-grey styling to match the project site
BG, GRID, INK, SERIES, ACCENT = "#1a1a1a", "#404040", "#c9c9c9", "#3987e5", "#d95926"


def apply_plot_style(plt):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 11,
    })


def load_wav(path):
    """Read a WAV file, mix to mono, normalize to full scale, remove DC.

    Returns (fs, x) with x float64 in [-1, 1] (so spectra are true dBFS).
    """
    fs, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x -= x.mean()
    return fs, x


def welch_db(x, fs, seg_seconds=4.0):
    """Welch PSD in dBFS/Hz. Segment length is shortened if x is too short."""
    nperseg = min(int(seg_seconds * fs), len(x))
    f, pxx = signal.welch(x, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=int(0.75 * nperseg), detrend="constant")
    return f, 10.0 * np.log10(np.maximum(pxx, 1e-30))


def tonal_peaks(f, db, fmax=FMAX_TONES, min_prom_db=MIN_PROMINENCE_DB):
    """Peaks whose level is >= min_prom_db above a median-filtered baseline.

    Returns (f_peak, level_db, prominence_db) rows, frequency-refined by
    parabolic interpolation on the dB spectrum, sorted by prominence.
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


@dataclass
class Calibration:
    """Microphone calibration: absolute sensitivity + response correction."""
    sens_dbfs: float            # dBFS produced by 94 dB SPL at 1 kHz
    freq: np.ndarray            # Hz, ascending
    gain: np.ndarray            # dB correction to add at freq

    @classmethod
    def load(cls, path):
        """Parse a miniDSP/REW-style cal file.

        Expects a 'Sens Factor =<x>dB' header; lines of 'freq gain' follow.
        Raises ValueError if no sensitivity header is found, since absolute
        SPL is then undefined.
        """
        sens = None
        freq, gain = [], []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                m = re.search(r"Sens\s*Factor\s*=\s*(-?\d+(?:\.\d+)?)\s*dB",
                              line, flags=re.IGNORECASE)
                if m:
                    sens = float(m.group(1))
                    continue
                parts = line.replace(",", " ").split()
                try:
                    fv, gv = float(parts[0]), float(parts[1])
                except (IndexError, ValueError):
                    continue  # comment / header line
                freq.append(fv)
                gain.append(gv)
        if sens is None:
            raise ValueError(f"{path}: no 'Sens Factor' header; "
                             "cannot convert dBFS to dB SPL")
        if not freq:
            raise ValueError(f"{path}: no frequency-response rows")
        order = np.argsort(freq)
        return cls(sens, np.asarray(freq)[order], np.asarray(gain)[order])

    def dbfs_to_spl(self, f, db):
        """Convert a dBFS spectrum (PSD or band level) to dB SPL."""
        corr = np.interp(f, self.freq, self.gain)
        return db + (94.0 - self.sens_dbfs) + corr


@dataclass
class Recording:
    """One manifest row: a recording with its position and calibration."""
    file: str                   # absolute path to the WAV
    tag: str
    x: float = None
    y: float = None
    z: float = None
    cal: Calibration = None
    notes: str = ""

    @property
    def calibrated(self):
        return self.cal is not None


def read_manifest(path):
    """Read a campaign manifest CSV.

    Columns: file, tag[, x, y, z, cal_file, notes]. Paths are resolved
    relative to the manifest's directory. Blank coordinates become None.
    cal_file is parsed once per distinct path.
    """
    base = os.path.dirname(os.path.abspath(path))
    cals = {}
    rows = []
    with open(path, newline="") as fh:
        for i, rec in enumerate(csv.DictReader(fh), start=2):
            if not (rec.get("file") or "").strip():
                continue
            def coord(key):
                v = (rec.get(key) or "").strip()
                return float(v) if v else None
            cal = None
            cal_file = (rec.get("cal_file") or "").strip()
            if cal_file:
                cal_path = os.path.join(base, cal_file)
                if cal_path not in cals:
                    cals[cal_path] = Calibration.load(cal_path)
                cal = cals[cal_path]
            tag = (rec.get("tag") or "").strip()
            if not tag:
                raise ValueError(f"{path}:{i}: missing tag")
            rows.append(Recording(
                file=os.path.join(base, rec["file"].strip()),
                tag=tag, x=coord("x"), y=coord("y"), z=coord("z"),
                cal=cal, notes=(rec.get("notes") or "").strip()))
    if not rows:
        raise ValueError(f"{path}: no recordings in manifest")
    return rows


def band_level_db(f, db, f0, half_bw=1.0):
    """Integrated level (dB) of the band f0 +/- half_bw from a PSD in dB/Hz.

    Power integration is robust to spectral leakage spreading a tone over
    a few bins, unlike reading the peak bin alone.
    """
    df = f[1] - f[0]
    sel = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    if not sel.any():
        return None
    return 10.0 * np.log10(np.sum(10.0 ** (db[sel] / 10.0)) * df + 1e-30)


def track_tone(x, fs, f0, half_bw=2.0, seg_seconds=2.0):
    """Track a tone's frequency over time for drift statistics.

    STFT with seg_seconds Hann windows at 50% overlap; in each frame the
    peak bin within f0 +/- half_bw is refined by parabolic interpolation.
    Returns (t, f_track) arrays; frames where the tone fell below the
    in-band median + 6 dB are dropped (tone absent / buried).
    """
    nperseg = min(int(seg_seconds * fs), len(x))
    f, t, sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg,
                                   noverlap=nperseg // 2, detrend="constant")
    sel = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    fb = f[sel]
    db = 10.0 * np.log10(np.maximum(sxx[sel, :], 1e-30))
    df = f[1] - f[0]
    times, freqs = [], []
    for j in range(db.shape[1]):
        col = db[:, j]
        i = int(np.argmax(col))
        if col[i] < np.median(col) + 6.0:
            continue
        if 0 < i < len(fb) - 1:
            a, b, c = col[i - 1], col[i], col[i + 1]
            delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        else:
            delta = 0.0
        times.append(t[j])
        freqs.append(fb[i] + delta * df)
    return np.asarray(times), np.asarray(freqs)


def drift_stats(t, f_track):
    """Summary statistics of a tone-frequency track.

    Returns dict with mean/std/min/max frequency (Hz) and the maximum
    drift rate (Hz/min) between consecutive frames, or None if the track
    has fewer than 2 frames.
    """
    if len(t) < 2:
        return None
    rates = np.abs(np.diff(f_track) / np.diff(t)) * 60.0
    return {"f_mean": float(np.mean(f_track)), "f_std": float(np.std(f_track)),
            "f_min": float(np.min(f_track)), "f_max": float(np.max(f_track)),
            "max_rate_hz_per_min": float(np.max(rates)),
            "n_frames": int(len(t)), "span_s": float(t[-1] - t[0])}
