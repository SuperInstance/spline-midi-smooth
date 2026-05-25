# spline-midi-smooth — Spline Interpolation for MIDI Automation

〰️ Bridge discrete CC events into smooth curves, eliminating zipper noise.

MIDI control changes are sparse, discrete points. When a DAW or synthesizer reads them, the jumps create audible **zipper noise** — a staircase artifact. spline-midi-smooth fits continuous spline curves through those points, then resamples at high density to produce smooth, artifact-free automation.

## Install

```bash
pip install spline-midi-smooth
```

Requires Python 3.10+, `numpy`, `mido`.

## Quick Start

### Interpolate sparse CC events

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
# Smoothed from 5 to 2000 samples
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

### Deadband guarantee

```python
from spline_midi_smooth.deadband_spline import deadband_spline, deadband_spline_exact_proof

# Smooth with deadband guarantee — curve stays within ±ε
times, values, upper, lower = deadband_spline(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8), (3, -0.1)],
    epsilon=1.0,
    smooth_method="cubic_hermite",
)

# Prove the linear spline stays inside the funnel
proven = deadband_spline_exact_proof(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8)],
    epsilon=1.0,
)
print(f"Deadband guarantee: {proven}")  # True
```

## Spline Methods

### Cubic Hermite

Passes through every point. Tangents estimated from neighbours via finite differences. C¹ continuous (smooth first derivative). Best for general-purpose automation where you need the curve to hit every control point.

```python
from spline_midi_smooth.interpolation import cubic_hermite

spline = cubic_hermite([(0, 0), (0.5, 80), (1.0, 100), (1.5, 90), (2.0, 64)])
```

### Catmull-Rom

Special case of Hermite: tangent at each point is `T_i = tension × (P_{i+1} − P_{i−1})`. The `tension` parameter controls tightness: 0.0 = linear, 0.5 = classic, 1.0 = loose. Best for velocity curves with intuitive shape control.

```python
from spline_midi_smooth.interpolation import catmull_rom

spline = catmull_rom([(0, 0), (1, 10), (2, 5)], tension=0.5)
```

### B-Spline

**Approximating** spline — does NOT pass through control points (except endpoints, clamped knot vector). Control points pull the curve toward them. Based on Cox-de Boor recursion. More numerically stable for many points. Best for smooth envelopes that approximate the general shape.

```python
from spline_midi_smooth.interpolation import bspline

spline = bspline([(0, 0), (1, 10), (2, 5), (3, 8)], degree=3)
```

## API Reference

### Interpolation Kernels

All accept a sequence of `(x, y)` tuples (x strictly increasing) and return a callable `spline_fn(x) → scalar or array`.

```python
from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom, bspline

spline = cubic_hermite(points)
spline = catmull_rom(points, tension=0.5)
spline = bspline(points, degree=3)
```

### MIDI Processing

```python
from spline_midi_smooth.midi_processor import smooth_midi_cc, smooth_midi_volume

# Smooth all CC tracks in a MIDI file
stats = smooth_midi_cc("in.mid", "out.mid", method="cubic_hermite", rate_hz=1000)

# Smooth only volume (CC#7)
n = smooth_midi_volume("in.mid", "out.mid", channel=0, rate_hz=1000)
```

### Anti-Aliasing

High-level wrappers for specific MIDI event types:

```python
from spline_midi_smooth.anti_alias import smooth_pitch_bend, smooth_velocity_curve, smooth_tempo_map

# Pitch bend at 2 kHz
n = smooth_pitch_bend("in.mid", "out.mid", rate_hz=2000, channel=0)

# Velocity curve — human-like dynamics
n = smooth_velocity_curve("in.mid", "out.mid", method="catmull_rom", rate_hz=500)

# Tempo map — smooth rubato
n = smooth_tempo_map("in.mid", "out.mid", method="cubic_hermite", rate_hz=10)
```

### Deadband Splines

Constraint-verified smoothing — proves the smoothed curve stays within ±ε of the original:

```python
from spline_midi_smooth.deadband_spline import deadband_bounds, deadband_spline, deadband_spline_exact_proof

# Compute deadband boundaries
upper, lower = deadband_bounds([0.0, 1.0, 2.0, 3.0], epsilon=5.0)

# Smooth with guarantee
times, values, upper, lower = deadband_spline(phase_offsets, epsilon=1.0, smooth_method="cubic_hermite")

# Exact proof for linear spline (convexity argument)
proven = deadband_spline_exact_proof(phase_offsets, epsilon=1.0)
```

The deadband guarantee: a degree-1 (piecewise-linear) spline with all control points in [−ε, +ε] stays within [−ε, +ε] everywhere, by convexity. This is the constraint-theory foundation.

## Architecture

```
spline_midi_smooth/
├── interpolation.py     # Core spline kernels (cubic_hermite, catmull_rom, bspline)
├── midi_processor.py    # MIDI file I/O: CC smoothing, volume smoothing
├── anti_alias.py        # High-level wrappers: pitch bend, velocity, tempo
└── deadband_spline.py   # Constraint verification: bounds, exact proof
```

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

## Related Repos

- [snapkit-v2](https://github.com/SuperInstance/snapkit-v2) — Eisenstein lattice snap + spectral analysis
- [style-dna](https://github.com/SuperInstance/style-dna) — Musical DNA extraction and style morphing
- [holonomy-harmony](https://github.com/SuperInstance/holonomy-harmony) — Chord progression analysis via holonomy
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — Mathematical primitives

## License

Apache License 2.0
