"""Tests for MIDI CC processor."""

import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest

from spline_midi_smooth.midi_processor import smooth_midi_cc, smooth_midi_volume


def _make_midi_with_cc(
    cc_events: list[tuple[float, int, int, int]], tpb=480
) -> Path:
    """Create temp MIDI with CC events.

    cc_events = [(sec, channel, control, value)]
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    tempo = 500000
    prev_tick = 0
    for sec, ch, ctrl, val in cc_events:
        tick = int(mido.second2tick(sec, tpb, tempo))
        track.append(
            mido.Message("control_change", channel=ch, control=ctrl, value=val,
                         time=tick - prev_tick)
        )
        prev_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(tmp.name)
    return Path(tmp.name)


class TestSmoothMidiCC:
    def test_basic_smoothing(self):
        """Smooth a CC track and verify output exists with dense events."""
        events = [(i * 0.5, 0, 1, int(64 + 30 * np.sin(i))) for i in range(8)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".out.mid")
        stats = smooth_midi_cc(in_path, out_path, rate_hz=100)
        assert out_path.exists()
        # Should have generated more events than input
        total = sum(stats.values())
        assert total > len(events)

    def test_output_values_in_range(self):
        events = [(i * 0.3, 0, 7, min(127, max(0, 64 + i * 10))) for i in range(6)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".out.mid")
        smooth_midi_cc(in_path, out_path, rate_hz=500)
        mid = mido.MidiFile(str(out_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type == "control_change":
                    assert 0 <= msg.value <= 127

    def test_all_methods(self):
        events = [(i * 0.2, 0, 1, 64 + i * 5) for i in range(6)]
        for method in ["cubic_hermite", "catmull_rom", "bspline"]:
            in_path = _make_midi_with_cc(events)
            out_path = in_path.with_suffix(f".{method}.mid")
            stats = smooth_midi_cc(in_path, out_path, method=method, rate_hz=100)
            assert out_path.exists()
            assert sum(stats.values()) >= 2

    def test_single_cc_point_passthrough(self):
        """Single CC point → not enough to interpolate, kept as-is."""
        in_path = _make_midi_with_cc([(0.0, 0, 1, 64)])
        out_path = in_path.with_suffix(".out.mid")
        stats = smooth_midi_cc(in_path, out_path)
        total = sum(stats.values())
        assert total == 1

    def test_keep_original_cc(self):
        events = [(0.0, 0, 1, 64), (1.0, 0, 1, 100)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".out.mid")
        stats = smooth_midi_cc(in_path, out_path, keep_original_cc=True, rate_hz=100)
        total = sum(stats.values())
        # With keep_original, total should be even larger
        assert total >= len(events)

    def test_unknown_method_raises(self):
        events = [(0.0, 0, 1, 64), (1.0, 0, 1, 100)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".out.mid")
        with pytest.raises(ValueError, match="Unknown spline method"):
            smooth_midi_cc(in_path, out_path, method="nonexistent")


class TestSmoothMidiVolume:
    def test_volume_smoothing(self):
        events = [(i * 0.3, 0, 7, 64 + i * 10) for i in range(5)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".vol.mid")
        n = smooth_midi_volume(in_path, out_path, rate_hz=100)
        assert out_path.exists()
        assert n >= 2

    def test_no_volume_returns_zero(self):
        """No CC#7 events → copy unchanged, return 0."""
        events = [(0.0, 0, 1, 64), (1.0, 0, 1, 100)]
        in_path = _make_midi_with_cc(events)
        out_path = in_path.with_suffix(".vol.mid")
        n = smooth_midi_volume(in_path, out_path)
        assert n == 0
