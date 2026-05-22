"""spline_midi_smooth — Spline-based MIDI smoothing with deadband theory."""

from .interpolation import cubic_hermite, catmull_rom, bspline
from .deadband_spline import deadband_bounds, deadband_spline, deadband_spline_exact_proof, is_deadband_a_spline
from .anti_alias import smooth_pitch_bend, smooth_velocity_curve, smooth_tempo_map
from .midi_processor import CcTrackKey, CcPoint, smooth_midi_cc, smooth_midi_volume

__version__ = "0.1.0"

__all__ = [
    "cubic_hermite",
    "catmull_rom",
    "bspline",
    "deadband_bounds",
    "deadband_spline",
    "deadband_spline_exact_proof",
    "is_deadband_a_spline",
    "smooth_pitch_bend",
    "smooth_velocity_curve",
    "smooth_tempo_map",
    "CcTrackKey",
    "CcPoint",
    "smooth_midi_cc",
    "smooth_midi_volume",
    "__version__",
]
