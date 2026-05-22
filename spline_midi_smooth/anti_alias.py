"""
Anti-aliasing via spline interpolation for MIDI expressive parameters.

Discrete parameter jumps in digital MIDI cause audible artifacts:
- Pitch bend "steps" (zipper noise) when coarse 14-bit values change
- Velocity quantization producing robotic dynamics
- Abrupt tempo changes destroying musical rubato

Splining each parameter domain converts discrete events into continuous
functions, exactly the technique used in high-quality wavetable
synthesis and analog-circuit emulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mido
import numpy as np

from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom
from spline_midi_smooth.midi_processor import _midi_to_absolute_seconds


# ---------------------------------------------------------------------------
# Pitch-bend anti-alias
# ---------------------------------------------------------------------------

def smooth_pitch_bend(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "cubic_hermite",
    rate_hz: float = 2000.0,
    channel: int | None = None,
) -> int:
    """Resample pitch-bend events at *rate_hz* using spline interpolation.

    MIDI pitch bend is a 14-bit signed value (-8192 … +8191).  Smoothing
    at 2 kHz yields a near-continuous pitch trajectory that eliminates
    zipper noise while remaining perfectly reconstructible from the
    original sparse events.

    Parameters
    ----------
    input_path, output_path : path-like
    method : str, default "cubic_hermite"
    rate_hz : float, default 2000
        Pitch-bend is the most audible parameter; oversample aggressively.
    channel : int or None
        If given, smooth only this channel's bend wheel.

    Returns
    -------
    n_generated : int
        Total number of pitch-bend events written.
    """
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)

    # collect bend events per channel
    bends: dict[int, list[tuple[float, int]]] = {}
    for sec, msg in events:
        if msg.type == "pitchwheel":
            if channel is not None and msg.channel != channel:
                continue
            bends.setdefault(msg.channel, []).append((sec, msg.pitch))

    if not bends:
        mid_in.save(str(output_path))
        return 0

    new_bends: list[tuple[float, int, int]] = []  # (sec, channel, pitch)
    total = 0
    for ch, pts in bends.items():
        if len(pts) < 2:
            for sec, val in pts:
                new_bends.append((sec, ch, val))
            total += len(pts)
            continue
        pts.sort(key=lambda t: t[0])
        spline_fn = cubic_hermite(pts) if method == "cubic_hermite" else catmull_rom(pts)
        t0, t1 = pts[0][0], pts[-1][0]
        n = max(2, int(np.ceil((t1 - t0) * rate_hz)))
        times = np.linspace(t0, t1, n)
        vals = spline_fn(times)
        vals = np.clip(np.rint(vals), -8192, 8191).astype(int)
        for t, v in zip(times, vals):
            new_bends.append((float(t), ch, int(v)))
        total += n

    # rebuild file
    mid_out = mido.MidiFile(type=0, ticks_per_beat=mid_in.ticks_per_beat)
    out_track = mido.MidiTrack()
    mid_out.tracks.append(out_track)

    other: list[tuple[float, mido.Message]] = []
    for sec, msg in events:
        if msg.type != "pitchwheel":
            other.append((sec, msg))
        elif channel is not None and msg.channel != channel:
            other.append((sec, msg))

    all_events = list(other)
    for sec, ch, val in new_bends:
        all_events.append((sec, mido.Message("pitchwheel", channel=ch, pitch=val)))
    all_events.sort(key=lambda t: t[0])

    prev_sec = 0.0
    default_tempo = 500000
    for sec, msg in all_events:
        dt = max(0, int(mido.second2tick(sec - prev_sec, mid_out.ticks_per_beat, default_tempo)))
        out_track.append(msg.copy(time=dt))
        prev_sec = sec
    out_track.append(mido.MetaMessage("end_of_track", time=0))
    mid_out.save(str(output_path))
    return total


# ---------------------------------------------------------------------------
# Velocity curve anti-alias
# ---------------------------------------------------------------------------

def smooth_velocity_curve(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "catmull_rom",
    rate_hz: float = 500.0,
    channel: int | None = None,
) -> int:
    """Spline-interpolate note-on velocities to create human-like dynamics.

    Typical MIDI editors quantise velocities to a few discrete steps
    (e.g. 64, 80, 100).  A Catmull-Rom spline through those points
    produces a continuously varying dynamic contour — the difference
    between a drum machine and a living performance.

    Rather than inserting new notes, this function replaces each
    existing note-on velocity with the spline-evaluated value at the
    exact note start time, then inserts intermediate note-on events
    (with corresponding note-offs) if the velocity contour changes
    significantly between adjacent notes.

    Parameters
    ----------
    input_path, output_path : path-like
    method : str, default "catmull_rom"
    rate_hz : float, default 500
        How many velocity samples per second to generate between notes.
    channel : int or None
        Restrict smoothing to a single MIDI channel.

    Returns
    -------
    n_generated : int
        Number of new (or rewritten) note-on events.
    """
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)

    # Gather note_on (velocity > 0) events per channel
    note_ons: dict[int, list[tuple[float, int, int]]] = {}  # ch -> [(sec, note, vel)]
    note_offs: dict[int, list[tuple[float, int]]] = {}      # ch -> [(sec, note)]
    for sec, msg in events:
        if msg.type == "note_on" and msg.velocity > 0:
            if channel is not None and msg.channel != channel:
                continue
            note_ons.setdefault(msg.channel, []).append((sec, msg.note, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if channel is not None and msg.channel != channel:
                continue
            note_offs.setdefault(msg.channel, []).append((sec, msg.note))

    if not note_ons:
        mid_in.save(str(output_path))
        return 0

    new_notes: list[tuple[float, mido.Message]] = []
    total = 0
    for ch, pts in note_ons.items():
        if len(pts) < 2:
            for sec, note, vel in pts:
                new_notes.append((sec, mido.Message("note_on", channel=ch, note=note, velocity=vel)))
            total += len(pts)
            continue

        pts.sort(key=lambda t: t[0])
        # build spline through (time, velocity)
        xy = [(p[0], float(p[2])) for p in pts]
        spline_fn = cubic_hermite(xy) if method == "cubic_hermite" else catmull_rom(xy)

        # sample at rate_hz between first and last note
        t0, t1 = pts[0][0], pts[-1][0]
        n = max(len(pts), int(np.ceil((t1 - t0) * rate_hz)))
        times = np.linspace(t0, t1, n)
        vels = spline_fn(times)
        vels = np.clip(np.rint(vels), 1, 127).astype(int)

        # Map each sample time to the note that is currently playing.
        # For simplicity we walk both lists in parallel.
        note_idx = 0
        active_note: int | None = None
        active_end: float = 0.0
        for t, v in zip(times, vels):
            # advance to the note covering time t
            while note_idx < len(pts) and pts[note_idx][0] <= t:
                active_note = pts[note_idx][1]
                # find matching note_off (naïve: next off for same note)
                off_time = t + 0.5  # fallback if no off found
                for osec, onote in note_offs.get(ch, []):
                    if onote == active_note and osec > pts[note_idx][0]:
                        off_time = osec
                        break
                active_end = off_time
                note_idx += 1

            if active_note is None:
                continue
            if t > active_end:
                continue

            new_notes.append((float(t), mido.Message("note_on", channel=ch, note=active_note, velocity=int(v))))
            total += 1

    # rebuild
    mid_out = mido.MidiFile(type=0, ticks_per_beat=mid_in.ticks_per_beat)
    out_track = mido.MidiTrack()
    mid_out.tracks.append(out_track)

    other: list[tuple[float, mido.Message]] = []
    for sec, msg in events:
        if msg.type == "note_on" and msg.velocity > 0:
            if channel is None or msg.channel == channel:
                continue
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if channel is None or msg.channel == channel:
                continue
        other.append((sec, msg))

    # add explicit note-offs for the new notes (reuse original off times)
    explicit_offs: list[tuple[float, mido.Message]] = []
    # We need note-off for every note-on we inserted.  Simplification:
    # keep all original note-offs (for the channels we are NOT touching)
    # and generate note-offs at original end times for touched channels.
    # To do this robustly, just keep original note-offs for the smoothed
    # channel and let overlapping note-ons create a legato re-articulation.
    for sec, msg in events:
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if channel is None or msg.channel == channel:
                explicit_offs.append((sec, msg))

    all_events = list(other)
    all_events.extend(new_notes)
    all_events.extend(explicit_offs)
    all_events.sort(key=lambda t: t[0])

    prev_sec = 0.0
    default_tempo = 500000
    for sec, msg in all_events:
        dt = max(0, int(mido.second2tick(sec - prev_sec, mid_out.ticks_per_beat, default_tempo)))
        out_track.append(msg.copy(time=dt))
        prev_sec = sec
    out_track.append(mido.MetaMessage("end_of_track", time=0))
    mid_out.save(str(output_path))
    return total


# ---------------------------------------------------------------------------
# Tempo map anti-alias (rubato)
# ---------------------------------------------------------------------------

def smooth_tempo_map(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "cubic_hermite",
    rate_hz: float = 10.0,
) -> int:
    """Convert discrete tempo-change events into a smooth rubato curve.

    Standard MIDI files carry tempo only via ``set_tempo`` meta events.
    Abrupt jumps (e.g. 120 BPM → 140 BPM) sound mechanical.  A spline
    through the tempo points yields a natural acceleration / ritardando
    exactly like a conductor's hand motion — itself a smooth spline.

    Parameters
    ----------
    input_path, output_path : path-like
    method : str, default "cubic_hermite"
    rate_hz : float, default 10
        Tempo changes are slow; 10 Hz is plenty for musical rubato.

    Returns
    -------
    n_generated : int
        Number of ``set_tempo`` events written.
    """
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)

    # collect set_tempo events
    tempos: list[tuple[float, int]] = []  # (sec, microseconds_per_quarter)
    for sec, msg in events:
        if msg.type == "set_tempo":
            tempos.append((sec, msg.tempo))

    if len(tempos) < 2:
        mid_in.save(str(output_path))
        return len(tempos)

    tempos.sort(key=lambda t: t[0])
    # We spline BPM for intuitive shape, then convert back to µsec/beat.
    # BPM = 60_000_000 / tempo
    xy = [(t[0], 60_000_000.0 / t[1]) for t in tempos]
    spline_fn = cubic_hermite(xy) if method == "cubic_hermite" else catmull_rom(xy)

    t0, t1 = xy[0][0], xy[-1][0]
    n = max(2, int(np.ceil((t1 - t0) * rate_hz)))
    times = np.linspace(t0, t1, n)
    bpms = spline_fn(times)
    # clamp to musically sane 10 … 300 BPM
    bpms = np.clip(bpms, 10.0, 300.0)
    micros = (60_000_000.0 / bpms).astype(int)

    new_tempos: list[tuple[float, mido.MetaMessage]] = []
    for t, us in zip(times, micros):
        new_tempos.append((float(t), mido.MetaMessage("set_tempo", tempo=int(us))))

    # rebuild
    mid_out = mido.MidiFile(type=0, ticks_per_beat=mid_in.ticks_per_beat)
    out_track = mido.MidiTrack()
    mid_out.tracks.append(out_track)

    other: list[tuple[float, mido.Message]] = []
    for sec, msg in events:
        if msg.type != "set_tempo":
            other.append((sec, msg))

    all_events = list(other)
    all_events.extend(new_tempos)
    all_events.sort(key=lambda t: t[0])

    prev_sec = 0.0
    # For tick conversion we need a reference tempo; use the first smoothed one
    ref_tempo = int(micros[0])
    for sec, msg in all_events:
        dt = max(0, int(mido.second2tick(sec - prev_sec, mid_out.ticks_per_beat, ref_tempo)))
        out_track.append(msg.copy(time=dt))
        prev_sec = sec
    out_track.append(mido.MetaMessage("end_of_track", time=0))
    mid_out.save(str(output_path))
    return n
