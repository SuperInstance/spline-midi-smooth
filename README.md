# spline-midi-smooth

〰️ Spline interpolation for MIDI automation — bridge discrete CC events into smooth curves, eliminating zipper noise.

MIDI control changes are sparse, discrete points. When a DAW or synthesizer reads them, the jumps between values create audible "zipper noise" — a staircase artifact. spline-midi-smooth fits continuous spline curves through those discrete points, then resamples at high density to produce smooth, artifact-free automation.

## Why it exists

Every MIDI producer has heard zipper noise: the audible stepping when a volume, filter cutoff, or pitch bend changes in coarse increments. The fix is mathematically straightforward — fit a smooth curve through the control points — but most DAWs don't give you fine-grained control over *which* spline to use or *how* to resample. This library gives you that control, plus a provable guarantee that the smoothed curve stays inside a deadband (the musical equivalent of "no overshoot").

## The math in plain English

**Cubic Hermite spline** — Passes through every point. Tangents are estimated automatically from neighbouring points using finite differences. C¹ continuous (smooth first derivative). Best for general-purpose automation where you need the curve to hit every control point.

**Catmull-Rom spline** — A special case of Hermite where the tangent at each point is derived from its neighbours: `T_i = tension × (P_{i+1} - P_{i-1})`. The `tension` parameter controls how tight the curve is: 0.0 = linear, 0.5 = classic, 1.0 = loose. Best for velocity curves where you want intuitive shape control.

**B-spline** — An *approximating* spline that does NOT pass through the control points (except first and last, due to clamped knot vector). Instead, the control points pull the curve toward them. Based on the Cox-de Boor recursive formula. More numerically stable for large numbers of points. Best for when you want a smooth envelope that approximates the general shape.

**Deadband spline** — A piecewise-linear spline constrained to stay within `[-ε, +ε]`. Proves that if all control points satisfy the deadband, the entire linear spline does too (convexity argument). This is the constraint-theory foundation: the spline is guaranteed to stay inside the funnel.

## Quick start

```bash
pip install spline-midi-smooth
```

```python
from spline_midi_smooth.interpolation import cubic_hermite
import numpy as np

# Sparse MIDI CC events: (time_seconds, value)
cc_points = [(0.0, 64), (0.5, 80), (1.0, 100), (1.5, 90), (2.0, 64)]

# Build smooth curve
spline = cubic_hermite(cc_points)

# Evaluate at any resolution
times = np.linspace(0.0, 2.0, 2000)  # 1 kHz
values = spline(times)
print(f"Smoothed from {len(cc_points)} to {len(values)} samples")
print(f"Value at t=0.75: {spline(0.75):.1f}")
```

Output:
```
Smoothed from 5 to 2000 samples
Value at t=0.75: 91.9
```

### Smooth a MIDI file

```python
from spline_midi_smooth.midi_processor import smooth_midi_cc

stats = smooth_midi_cc(
    input_path="input.mid",
    output_path="output.mid",
    method="cubic_hermite",  # or "catmull_rom" or "bspline"
    rate_hz=1000.0,           # 1000 CC events per second
)
for key, count in stats.items():
    print(f"CH{key.channel} CC#{key.control}: {count} events generated")
```

### Anti-alias pitch bend

```python
from spline_midi_smooth.anti_alias import smooth_pitch_bend

n = smooth_pitch_bend("input.mid", "output.mid", rate_hz=2000)
print(f"Generated {n} pitch bend events")
```

## API overview

### Interpolation kernels

```python
from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom, bspline

# All accept: sequence of (x, y) tuples, x strictly increasing
# All return: callable spline_fn(x) -> scalar or array

spline = cubic_hermite([(0, 0), (1, 10), (2, 5)])
spline = catmull_rom([(0, 0), (1, 10), (2, 5)], tension=0.5)
spline = bspline([(0, 0), (1, 10), (2, 5), (3, 8)], degree=3)
```

### MIDI processing

```python
from spline_midi_smooth.midi_processor import smooth_midi_cc, smooth_midi_volume

# Smooth all CC tracks
stats = smooth_midi_cc("in.mid", "out.mid", method="cubic_hermite", rate_hz=1000)

# Smooth only volume (CC#7)
n = smooth_midi_volume("in.mid", "out.mid", channel=0, rate_hz=1000)
```

### Anti-aliasing

```python
from spline_midi_smooth.anti_alias import smooth_pitch_bend, smooth_velocity_curve, smooth_tempo_map

# Pitch bend at 2 kHz
n = smooth_pitch_bend("in.mid", "out.mid", rate_hz=2000, channel=0)

# Velocity curve for human-like dynamics
n = smooth_velocity_curve("in.mid", "out.mid", method="catmull_rom", rate_hz=500)

# Tempo map for smooth rubato
n = smooth_tempo_map("in.mid", "out.mid", method="cubic_hermite", rate_hz=10)
```

### Deadband splines

```python
from spline_midi_smooth.deadband_spline import deadband_bounds, deadband_spline, deadband_spline_exact_proof

# Compute deadband boundaries
upper, lower = deadband_bounds([0.0, 1.0, 2.0, 3.0], epsilon=5.0)

# Smooth with deadband guarantee
times, values, upper, lower = deadband_spline(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8), (3, -0.1)],
    epsilon=1.0,
    smooth_method="cubic_hermite",
)

# Exact proof for linear spline
proven = deadband_spline_exact_proof(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8)],
    epsilon=1.0,
)
print(f"Deadband guarantee: {proven}")  # True
```

## Architecture

```
┌──────────────────┐
│  interpolation.py │  Core spline kernels
│  cubic_hermite    │  C¹ through points, auto tangents
│  catmull_rom      │  Interpolating, tension control
│  bspline          │  Approximating, Cox-de Boor
└────────┬─────────┘
         │ used by
    ┌────┴────────────────┐
    │                     │
┌───▼──────────┐  ┌───────▼───────┐
│midi_processor│  │  anti_alias   │
│smooth_midi_cc│  │ smooth_pitch  │
│smooth_volume │  │ smooth_vel    │
│resample CC   │  │ smooth_tempo  │
└──────────────┘  └───────────────┘
         │
    ┌────▼──────────┐
    │deadband_spline│  Constraint verification
    │bounds         │  Piecewise-linear proof
    │exact_proof    │  Deadband = degree-1 spline
    └───────────────┘
```

## Documentation

- [User Guide](docs/USER-GUIDE.md) — Complete usage documentation
- [Developer Guide](docs/DEVELOPER-GUIDE.md) — Contributing and internals
- [Examples](examples/) — Pitch bend and volume smoothing demos

## Related repos

- [holonomy-harmony](https://github.com/SuperInstance/holonomy-harmony) — Chord progression analysis via holonomy
- [plato-room-musician](https://github.com/SuperInstance/plato-room-musician) — PLATO rooms → MIDI music
- [tensor-midi](https://github.com/SuperInstance/tensor-midi) — INT8-saturated MIDI for neural synthesis

## Requirements

- Python 3.10+
- numpy
- mido (for MIDI file I/O)

## Install

```bash
pip install spline-midi-smooth
```

Or from source:

```bash
git clone https://github.com/SuperInstance/spline-midi-smooth.git
cd spline-midi-smooth
pip install -e .
```

## License

Apache License 2.0
