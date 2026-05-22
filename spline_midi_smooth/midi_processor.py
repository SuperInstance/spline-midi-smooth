"""
MIDI CC automation processor.

Reads a MIDI file, isolates control-change tracks per (channel, cc),
applies spline interpolation, and writes a new MIDI file where the
sparse CC events are replaced by a densely sampled smooth curve.

This proves the thesis that splines bridge discrete MIDI events and
continuous automation curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import mido
import numpy as np

from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom, bspline


@dataclass(frozen=True)
class CcTrackKey:
    channel: int
    control: int


@dataclass
class CcPoint:
    time_sec: float
    value: int  # 0–127


# ---------------------------------------------------------------------------
# MIDI parsing helpers
# ---------------------------------------------------------------------------

def _midi_to_absolute_seconds(
    mid: mido.MidiFile,
) -> list[tuple[float, mido.Message]]:
    """Return list of (absolute_time_sec, message) for every track event."""
    events: list[tuple[float, mido.Message]] = []
    for track in mid.tracks:
        tempo = 500000  # default 120 BPM (microseconds per quarter note)
        ticks = 0
        for msg in track:
            ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            # convert ticks → seconds using current tempo
            sec = mido.tick2second(ticks, mid.ticks_per_beat, tempo)
            events.append((sec, msg))
    return events


def _collect_cc_tracks(
    events: list[tuple[float, mido.Message]],
) -> dict[CcTrackKey, list[CcPoint]]:
    """Group control_change messages by (channel, control_number)."""
    tracks: dict[CcTrackKey, list[CcPoint]] = {}
    for sec, msg in events:
        if msg.type != "control_change":
            continue
        key = CcTrackKey(channel=msg.channel, control=msg.control)
        tracks.setdefault(key, []).append(CcPoint(time_sec=sec, value=msg.value))
    # sort by time (they usually are, but be safe)
    for key in tracks:
        tracks[key].sort(key=lambda p: p.time_sec)
    return tracks


# ---------------------------------------------------------------------------
# Spline choice dispatcher
# ---------------------------------------------------------------------------

def _make_spline(
    points: list[CcPoint],
    method: str = "cubic_hermite",
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Build a 1-D spline from CC points."""
    xy = [(p.time_sec, float(p.value)) for p in points]
    if len(xy) < 2:
        # degenerate: constant value
        val = float(xy[0][1]) if xy else 64.0
        return lambda x: np.full_like(np.atleast_1d(x), val, dtype=float)
    if method == "cubic_hermite":
        return cubic_hermite(xy)
    if method == "catmull_rom":
        return catmull_rom(xy)
    if method == "bspline":
        return bspline(xy, degree=3)
    raise ValueError(f"Unknown spline method: {method}")


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_cc(
    points: list[CcPoint],
    spline_fn: Callable[[float | np.ndarray], float | np.ndarray],
    rate_hz: float,
    value_min: int = 0,
    value_max: int = 127,
) -> list[CcPoint]:
    """Evaluate a spline at *rate_hz* samples/sec across the span of *points*.

    Returns a dense list of CC points.  Values are clamped to MIDI range.
    """
    if not points:
        return []
    t0 = points[0].time_sec
    t1 = points[-1].time_sec
    duration = t1 - t0
    if duration <= 0:
        return [CcPoint(t0, int(np.clip(points[0].value, value_min, value_max)))]

    n_samples = max(2, int(np.ceil(duration * rate_hz)))
    times = np.linspace(t0, t1, n_samples)
    values = spline_fn(times)
    values = np.clip(np.rint(values), value_min, value_max).astype(int)
    return [CcPoint(float(t), int(v)) for t, v in zip(times, values)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_midi_cc(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "cubic_hermite",
    rate_hz: float = 1000.0,
    keep_original_cc: bool = False,
) -> dict[CcTrackKey, int]:
    """Read a MIDI file, spline-smooth every CC automation track, and write result.

    Parameters
    ----------
    input_path : path-like
        Source MIDI file.
    output_path : path-like
        Destination MIDI file.
    method : {"cubic_hermite", "catmull_rom", "bspline"}, default "cubic_hermite"
        Which spline kernel to use between CC control points.
    rate_hz : float, default 1000.0
        Output sampling rate for the smoothed CC curves (events per second).
    keep_original_cc : bool, default False
        If True, original CC events are retained alongside the dense ones.

    Returns
    -------
    stats : dict[CcTrackKey, int]
        Mapping from each smoothed CC track to the number of generated events.
    """
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)
    cc_tracks = _collect_cc_tracks(events)

    stats: dict[CcTrackKey, int] = {}
    new_cc_events: list[tuple[float, CcTrackKey, int]] = []

    for key, points in cc_tracks.items():
        if len(points) < 2:
            # Not enough points to interpolate — keep original
            for p in points:
                new_cc_events.append((p.time_sec, key, p.value))
            stats[key] = len(points)
            continue

        spline_fn = _make_spline(points, method=method)
        dense = _resample_cc(points, spline_fn, rate_hz=rate_hz)
        for p in dense:
            new_cc_events.append((p.time_sec, key, p.value))
        stats[key] = len(dense)

        if keep_original_cc:
            for p in points:
                new_cc_events.append((p.time_sec, key, p.value))

    # rebuild a single-track MIDI file (type 0) for maximum compatibility
    mid_out = mido.MidiFile(type=0, ticks_per_beat=mid_in.ticks_per_beat)
    out_track = mido.MidiTrack()
    mid_out.tracks.append(out_track)

    # Gather non-CC meta events from original (tempo, time signature, etc.)
    meta_events: list[tuple[float, mido.Message]] = []
    for sec, msg in events:
        if msg.is_meta:
            meta_events.append((sec, msg))
        elif msg.type != "control_change":
            # Keep non-CC channel messages (notes, pitch bend, etc.)
            meta_events.append((sec, msg))

    # Merge everything, sort by time, then write with delta ticks
    all_events = list(meta_events)
    for sec, key, val in new_cc_events:
        all_events.append(
            (sec, mido.Message("control_change", channel=key.channel,
                               control=key.control, value=val))
        )

    # stable sort by time, preserving meta-before-note ordering where helpful
    all_events.sort(key=lambda t: t[0])

    # Simplest approach: write everything to one track using absolute→delta
    prev_sec = 0.0
    # default tempo for tick conversion
    default_tempo = 500000
    for sec, msg in all_events:
        delta_sec = sec - prev_sec
        delta_ticks = int(mido.second2tick(delta_sec, mid_out.ticks_per_beat, default_tempo))
        delta_ticks = max(0, delta_ticks)
        msg_copy = msg.copy(time=delta_ticks)
        out_track.append(msg_copy)
        prev_sec = sec

    # end-of-track meta
    out_track.append(mido.MetaMessage("end_of_track", time=0))

    mid_out.save(str(output_path))
    return stats


# ---------------------------------------------------------------------------
# Convenience wrappers for specific use-cases
# ---------------------------------------------------------------------------

def smooth_midi_volume(
    input_path: str | Path,
    output_path: str | Path,
    *,
    channel: int | None = None,
    method: str = "cubic_hermite",
    rate_hz: float = 1000.0,
) -> int:
    """Smooth only CC #7 (volume) on an optional specific channel."""
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)
    cc_tracks = _collect_cc_tracks(events)

    target = CcTrackKey(channel=channel if channel is not None else 0, control=7)
    candidates = {k: v for k, v in cc_tracks.items() if k.control == 7}
    if channel is not None:
        candidates = {k: v for k, v in candidates.items() if k.channel == channel}

    if not candidates:
        # No volume automation found — copy file unchanged
        mid_in.save(str(output_path))
        return 0

    # Proceed with standard pipeline but only for volume tracks
    new_cc_events: list[tuple[float, CcTrackKey, int]] = []
    total = 0
    for key, points in candidates.items():
        spline_fn = _make_spline(points, method=method)
        dense = _resample_cc(points, spline_fn, rate_hz=rate_hz)
        for p in dense:
            new_cc_events.append((p.time_sec, key, p.value))
        total += len(dense)

    mid_out = mido.MidiFile(type=0, ticks_per_beat=mid_in.ticks_per_beat)
    out_track = mido.MidiTrack()
    mid_out.tracks.append(out_track)

    meta_and_notes: list[tuple[float, mido.Message]] = []
    for sec, msg in events:
        if msg.is_meta or (not msg.type == "control_change"):
            meta_and_notes.append((sec, msg))
        elif msg.type == "control_change" and msg.control != 7:
            meta_and_notes.append((sec, msg))

    all_events = list(meta_and_notes)
    for sec, key, val in new_cc_events:
        all_events.append(
            (sec, mido.Message("control_change", channel=key.channel,
                               control=key.control, value=val))
        )
    all_events.sort(key=lambda t: t[0])

    prev_sec = 0.0
    default_tempo = 500000
    for sec, msg in all_events:
        delta_sec = sec - prev_sec
        delta_ticks = int(mido.second2tick(delta_sec, mid_out.ticks_per_beat, default_tempo))
        delta_ticks = max(0, delta_ticks)
        out_track.append(msg.copy(time=delta_ticks))
        prev_sec = sec

    out_track.append(mido.MetaMessage("end_of_track", time=0))
    mid_out.save(str(output_path))
    return total
