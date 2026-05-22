"""Tests for anti-alias module (pitch bend, velocity, tempo smoothing)."""

import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest

from spline_midi_smooth.anti_alias import (
    smooth_pitch_bend,
    smooth_velocity_curve,
    smooth_tempo_map,
)


def _make_midi_with_pitchbend(bends: list[tuple[float, int]], tpb=480) -> Path:
    """Create a temp MIDI file with pitchwheel events."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    tempo = 500000
    prev_tick = 0
    for sec, pitch in bends:
        tick = int(mido.second2tick(sec, tpb, tempo))
        track.append(mido.Message("pitchwheel", pitch=pitch, time=tick - prev_tick))
        prev_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(tmp.name)
    return Path(tmp.name)


def _make_midi_with_notes(notes: list[tuple[float, int, int]], tpb=480) -> Path:
    """Create a temp MIDI with note_on events. notes = [(sec, note, velocity)]."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    tempo = 500000
    prev_tick = 0
    for sec, note, vel in notes:
        tick = int(mido.second2tick(sec, tpb, tempo))
        track.append(mido.Message("note_on", note=note, velocity=vel, time=tick - prev_tick))
        prev_tick = tick
        # note_off after 0.3s
        off_tick = int(mido.second2tick(sec + 0.3, tpb, tempo))
        track.append(mido.Message("note_off", note=note, velocity=0, time=off_tick - tick))
        prev_tick = off_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(tmp.name)
    return Path(tmp.name)


def _make_midi_with_tempos(tempos: list[tuple[float, int]], tpb=480) -> Path:
    """Create a temp MIDI with set_tempo events. tempos = [(sec, microseconds_per_beat)]."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    prev_tick = 0
    for sec, tempo in tempos:
        tick = int(mido.second2tick(sec, tpb, 500000))
        if tempo != 500000:
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - prev_tick))
        else:
            track.append(mido.Message("note_on", note=60, velocity=64, time=tick - prev_tick))
            track.append(mido.Message("note_off", note=60, velocity=0, time=10))
        prev_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(tmp.name)
    return Path(tmp.name)


class TestSmoothPitchBend:
    def test_produces_output_file(self):
        in_path = _make_midi_with_pitchbend([(0.0, 0), (1.0, 2000), (2.0, -1000)])
        out_path = in_path.with_suffix(".out.mid")
        n = smooth_pitch_bend(in_path, out_path)
        assert out_path.exists()
        assert n >= 2

    def test_values_in_valid_range(self):
        in_path = _make_midi_with_pitchbend([(0.0, 0), (0.5, 4000), (1.0, -3000)])
        out_path = in_path.with_suffix(".out.mid")
        smooth_pitch_bend(in_path, out_path)
        mid = mido.MidiFile(str(out_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type == "pitchwheel":
                    assert -8192 <= msg.pitch <= 8191

    def test_mean_preserved(self):
        bends = [(0.0, 0), (0.5, 2000), (1.0, 0)]
        in_path = _make_midi_with_pitchbend(bends)
        out_path = in_path.with_suffix(".out.mid")
        smooth_pitch_bend(in_path, out_path, rate_hz=500)
        mid = mido.MidiFile(str(out_path))
        vals = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == "pitchwheel":
                    vals.append(msg.pitch)
        if vals:
            orig_mean = np.mean([b[1] for b in bends])
            new_mean = np.mean(vals)
            assert abs(new_mean - orig_mean) < 1000

    def test_no_bends_returns_zero(self):
        """MIDI file with no pitch bends → 0 events generated."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)
        mid.save(tmp.name)
        out_path = tmp.name.replace(".mid", ".out.mid")
        n = smooth_pitch_bend(tmp.name, out_path)
        assert n == 0


class TestSmoothVelocity:
    def test_produces_output(self):
        notes = [(0.0, 60, 80), (0.5, 62, 100), (1.0, 64, 60)]
        in_path = _make_midi_with_notes(notes)
        out_path = in_path.with_suffix(".out.mid")
        n = smooth_velocity_curve(in_path, out_path)
        assert out_path.exists()
        assert n >= 3  # at least the original notes

    def test_velocities_in_valid_range(self):
        notes = [(i * 0.3, 60 + i, 20 + i * 20) for i in range(6)]
        in_path = _make_midi_with_notes(notes)
        out_path = in_path.with_suffix(".out.mid")
        smooth_velocity_curve(in_path, out_path)
        mid = mido.MidiFile(str(out_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    assert 1 <= msg.velocity <= 127


class TestSmoothTempo:
    def test_produces_output(self):
        tempos = [(0.0, 500000), (1.0, 400000), (2.0, 600000)]
        in_path = _make_midi_with_tempos(tempos)
        out_path = in_path.with_suffix(".out.mid")
        n = smooth_tempo_map(in_path, out_path)
        assert out_path.exists()
        assert n >= 2

    def test_tempos_in_sane_range(self):
        tempos = [(0.0, 500000), (1.0, 300000), (2.0, 700000)]
        in_path = _make_midi_with_tempos(tempos)
        out_path = in_path.with_suffix(".out.mid")
        smooth_tempo_map(in_path, out_path)
        mid = mido.MidiFile(str(out_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    bpm = 60_000_000 / msg.tempo
                    assert 10 <= bpm <= 300
