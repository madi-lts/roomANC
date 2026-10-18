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

# Intermittent-source gating (procedure, Phase 1 protocol): the HVAC and
# pumps cycle on ~30 s - 1 min timescales, and each machine cycles
# independently, so the on/off state is classified per tone, per frame.
GATE_SEG_SECONDS = 2.0   # frame length for the on/off classifier
GATE_SPLIT_DB = 6.0      # min P90-P10 frame-level spread to call it cycling
MIN_ON_SECONDS = 20.0    # per point/tone: cumulative on-time to call it done
MIN_ON_INTERVALS = 2     # per point/tone: distinct on-intervals to call it done

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


def frame_spectrogram(x, fs, seg_seconds=GATE_SEG_SECONDS):
    """Spectrogram with seg_seconds Hann frames at 50% overlap.

    Returns (f, t, sxx) with sxx a PSD (power/Hz); shared by the per-tone
    gates of one recording so the STFT runs once.
    """
    nperseg = min(int(seg_seconds * fs), len(x))
    return signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend="constant")


def _erode(on):
    """Shrink every on-run by one frame at each edge.

    Used when a gate from one device masks frames of another: the two
    recordings are started together only to ~1 s, so boundary frames of an
    on-interval may straddle the transition on the other device.
    """
    e = on.copy()
    e[1:] &= on[:-1]
    e[:-1] &= on[1:]
    return e


@dataclass
class ToneGate:
    """Per-frame on/off classification of one tone in one recording.

    The in-band level of each frame is compared against a threshold halfway
    between the 10th and 90th percentile frame levels; if those percentiles
    are closer than min_split_db the recording shows no off state for this
    tone ('steady': either the source never cycled, or the mic sits on a
    null where the on/off contrast vanishes — a fixed reference channel
    disambiguates the two).
    """
    f0: float
    t: np.ndarray          # frame centres (s)
    level_db: np.ndarray   # in-band level per frame (dBFS)
    on: np.ndarray         # bool: frame classified source-on
    hop_s: float           # frame hop (s)
    split_db: float        # P90 - P10 frame-level spread

    @property
    def steady(self):
        return bool(self.on.all())

    @property
    def on_fraction(self):
        return float(self.on.mean()) if len(self.on) else 0.0

    @property
    def on_seconds(self):
        return float(self.on.sum()) * self.hop_s

    @property
    def n_on_intervals(self):
        if not len(self.on):
            return 0
        return int(self.on[0]) + int(np.sum(self.on[1:] & ~self.on[:-1]))

    def level_over(self, mask):
        """Power-mean in-band level (dB) over the masked frames, or None."""
        if mask is None or not mask.any():
            return None
        p = 10.0 ** (self.level_db[mask] / 10.0)
        return float(10.0 * np.log10(np.mean(p) + 1e-30))

    @property
    def level_on_db(self):
        return self.level_over(self.on)

    @property
    def level_off_db(self):
        return None if self.steady else self.level_over(~self.on)

    def mask_at(self, t_other):
        """This gate's on-mask resampled at another recording's frame times.

        Edges are eroded by one frame first, absorbing the ~1 s start
        misalignment between two devices recording the same interval
        (band levels need no sample sync, only state agreement).
        """
        if len(self.t) < 2:
            return np.zeros(len(t_other), dtype=bool)
        e = _erode(self.on).astype(float)
        return np.interp(t_other, self.t, e) > 0.99

    def done(self, min_on_s=MIN_ON_SECONDS, min_intervals=MIN_ON_INTERVALS):
        """Has this recording captured enough on-state for this tone?"""
        return (self.on_seconds >= min_on_s
                and self.n_on_intervals >= min_intervals)


def tone_gate(x, fs, f0, half_bw=1.0, seg_seconds=GATE_SEG_SECONDS,
              min_split_db=GATE_SPLIT_DB, spec=None):
    """Classify each frame of x as source-on/off for the tone at f0.

    spec, if given, is a precomputed frame_spectrogram(x, fs) to share
    across the tones of one recording. Returns a ToneGate.
    """
    f, t, sxx = spec if spec is not None else frame_spectrogram(x, fs, seg_seconds)
    sel = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    df = f[1] - f[0]
    p = np.maximum(sxx[sel, :].sum(axis=0) * df, 1e-30)
    level = 10.0 * np.log10(p)
    hop_s = float(t[1] - t[0]) if len(t) > 1 else seg_seconds / 2.0
    p10, p90 = np.percentile(level, [10.0, 90.0])
    split = float(p90 - p10)
    if split < min_split_db:
        on = np.ones(len(level), dtype=bool)
    else:
        on = level >= 0.5 * (p10 + p90)
    return ToneGate(f0=f0, t=t, level_db=level, on=on, hop_s=hop_s,
                    split_db=split)


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
    ref_file: str = None        # simultaneous fixed-reference WAV, if any
    notes: str = ""

    @property
    def calibrated(self):
        return self.cal is not None


def read_manifest(path):
    """Read a campaign manifest CSV.

    Columns: file, tag[, x, y, z, cal_file, ref_file, notes]. Paths are
    resolved relative to the manifest's directory. Blank coordinates become
    None. cal_file is parsed once per distinct path. ref_file names the
    fixed reference channel recorded simultaneously with this row's file
    (intermittent-source protocol); blank means no reference.
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
            ref = (rec.get("ref_file") or "").strip()
            rows.append(Recording(
                file=os.path.join(base, rec["file"].strip()),
                tag=tag, x=coord("x"), y=coord("y"), z=coord("z"),
                cal=cal,
                ref_file=os.path.join(base, ref) if ref else None,
                notes=(rec.get("notes") or "").strip()))
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


def track_tone(x, fs, f0, half_bw=2.0, seg_seconds=2.0, gate=None):
    """Track a tone's frequency over time for drift statistics.

    STFT with seg_seconds Hann windows at 50% overlap; in each frame the
    peak bin within f0 +/- half_bw is refined by parabolic interpolation.
    Returns (t, f_track) arrays; frames where the tone fell below the
    in-band median + 6 dB are dropped (tone absent / buried). gate, a
    ToneGate for this tone, additionally drops source-off frames and the
    on/off transition frames (whose smeared spectra give spurious
    frequency estimates — essential for an intermittent source).
    """
    nperseg = min(int(seg_seconds * fs), len(x))
    f, t, sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg,
                                   noverlap=nperseg // 2, detrend="constant")
    sel = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    fb = f[sel]
    db = 10.0 * np.log10(np.maximum(sxx[sel, :], 1e-30))
    df = f[1] - f[0]
    on = None if gate is None else gate.mask_at(t)
    times, freqs = [], []
    for j in range(db.shape[1]):
        if on is not None and not on[j]:
            continue
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
