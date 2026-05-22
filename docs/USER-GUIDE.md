# User Guide — spline-midi-smooth

## Table of Contents

1. [Overview](#overview)
2. [Spline types](#spline-types)
3. [Interpolation API](#interpolation-api)
4. [MIDI CC smoothing](#midi-cc-smoothing)
5. [Pitch bend anti-aliasing](#pitch-bend-anti-aliasing)
6. [Velocity curve smoothing](#velocity-curve-smoothing)
7. [Tempo map smoothing](#tempo-map-smoothing)
8. [Deadband splines](#deadband-splines)
9. [Input/output formats](#inputoutput-formats)
10. [Configuration reference](#configuration-reference)
11. [Use cases](#use-cases)
12. [Troubleshooting](#troubleshooting)

## Overview

spline-midi-smooth converts sparse, discrete MIDI automation data into smooth, continuous curves. Every DAW eventually faces the problem: you draw a few CC points, and the synthesizer steps between them audibly. This library solves it by fitting proper spline curves through the control points and resampling at high density.

The pipeline is always: **parse MIDI → extract control points → fit spline → resample → write MIDI**.

## Spline types

### Cubic Hermite

```python
from spline_midi_smooth.interpolation import cubic_hermite

spline = cubic_hermite([(0, 64), (0.5, 100), (1.0, 80)])
```

- **Passes through every point**: Yes
- **Continuity**: C¹ (smooth first derivative)
- **Tangent estimation**: Automatic via central finite differences
- **Overshoot**: Possible (Gibbs-like phenomenon at sharp corners)
- **Best for**: General-purpose CC automation, filter sweeps

### Catmull-Rom

```python
from spline_midi_smooth.interpolation import catmull_rom

spline = catmull_rom([(0, 64), (0.5, 100), (1.0, 80)], tension=0.5)
```

- **Passes through every point**: Yes
- **Continuity**: C¹
- **Tangent estimation**: Derived from neighbours: `T_i = tension × (P_{i+1} - P_{i-1})`
- **Tension parameter**: 0.0 = linear, 0.5 = classic, 1.0 = loose
- **Best for**: Velocity curves, dynamics shaping

### B-spline

```python
from spline_midi_smooth.interpolation import bspline

spline = bspline([(0, 0), (0.3, 20), (0.6, 50), (1.0, 80)], degree=3)
```

- **Passes through every point**: No (approximating; first and last points are interpolated due to clamped knots)
- **Continuity**: C^(degree-1) for degree ≥ 2
- **Basis**: Cox-de Boor recursive formula
- **Degree**: Default 3 (cubic). Need ≥ degree+1 control points.
- **Best for**: Envelope shaping, general trends without hitting every point

## Interpolation API

All spline functions share the same interface:

```python
spline_fn = cubic_hermite(points)    # or catmull_rom, bspline
result = spline_fn(0.5)              # scalar
result = spline_fn(np.linspace(0,1)) # array
```

**Input**: `Sequence[tuple[float, float]]` — at least 2 points, x strictly increasing.

**Output**: `callable(x) -> float | np.ndarray`. Values outside domain are clamped to the nearest endpoint.

### Error handling

```python
# Too few points
cubic_hermite([(0, 1)])       # ValueError: requires at least 2 points

# Non-increasing x
cubic_hermite([(0, 1), (0, 2)])  # ValueError: x must be strictly increasing

# B-spline with too few points
bspline([(0, 1), (1, 2)], degree=3)  # ValueError: needs at least 4 points for degree 3
```

## MIDI CC smoothing

### `smooth_midi_cc` — Full-file CC smoothing

```python
from spline_midi_smooth.midi_processor import smooth_midi_cc

stats = smooth_midi_cc(
    input_path="song.mid",
    output_path="song_smooth.mid",
    method="cubic_hermite",     # "cubic_hermite" | "catmull_rom" | "bspline"
    rate_hz=1000.0,             # events per second per CC track
    keep_original_cc=False,     # keep original sparse events alongside
)
```

Returns `dict[CcTrackKey, int]` — number of generated events per (channel, CC#).

`CcTrackKey` is a frozen dataclass:

```python
@dataclass(frozen=True)
class CcTrackKey:
    channel: int
    control: int
```

### `smooth_midi_volume` — Volume-only smoothing

```python
from spline_midi_smooth.midi_processor import smooth_midi_volume

n = smooth_midi_volume(
    "input.mid", "output.mid",
    channel=0,              # None = all channels
    method="cubic_hermite",
    rate_hz=1000.0,
)
# Returns: number of generated CC#7 events
```

### What gets smoothed

`smooth_midi_cc` processes ALL control_change events in the file. It preserves:
- All meta events (tempo, time signature, track names)
- All note_on/note_off events
- All pitch_wheel events
- All program_change events

Only CC events are replaced by the dense resampled versions.

### Output format

Output is a Type-0 (single-track) MIDI file for maximum compatibility. All events from all tracks are merged into one timeline.

## Pitch bend anti-aliasing

```python
from spline_midi_smooth.anti_alias import smooth_pitch_bend

n = smooth_pitch_bend(
    "input.mid", "output.mid",
    method="cubic_hermite",    # only hermite and catmull_rom supported
    rate_hz=2000.0,            # pitch bend is audible — use high rate
    channel=None,              # None = all channels
)
```

Pitch bend is a 14-bit signed value (−8192 to +8191). At 2 kHz resampling, you get near-continuous pitch trajectories that eliminate zipper noise.

## Velocity curve smoothing

```python
from spline_midi_smooth.anti_alias import smooth_velocity_curve

n = smooth_velocity_curve(
    "input.mid", "output.mid",
    method="catmull_rom",      # default — good for dynamics
    rate_hz=500.0,             # velocity changes are slower than pitch
    channel=None,              # None = all channels
)
```

This function creates intermediate note-on events between existing notes, sampling the velocity curve at `rate_hz`. The result is a continuously varying dynamic contour — the difference between a drum machine and a human performance.

**Note**: This inserts additional note events. The original note-offs are preserved to maintain timing.

## Tempo map smoothing

```python
from spline_midi_smooth.anti_alias import smooth_tempo_map

n = smooth_tempo_map(
    "input.mid", "output.mid",
    method="cubic_hermite",
    rate_hz=10.0,              # tempo changes are slow — 10 Hz is plenty
)
```

Converts discrete tempo-change events into smooth accelerando/ritardando curves. Internally splines in BPM space, then converts back to microseconds-per-quarter-note for MIDI.

Output is clamped to 10–300 BPM (musically sane range).

## Deadband splines

The deadband module proves a constraint-theory result: a piecewise-linear spline through points inside a tolerance corridor stays inside the corridor.

### `deadband_bounds` — Compute the funnel

```python
from spline_midi_smooth.deadband_spline import deadband_bounds

# Uniform deadband
upper, lower = deadband_bounds([0.0, 1.0, 2.0, 3.0], epsilon=5.0)
# upper = [5, 5, 5, 5], lower = [-5, -5, -5, -5]

# Adaptive deadband (narrows over time)
upper, lower = deadband_bounds([0.0, 1.0, 2.0, 3.0], epsilon=[5.0, 4.0, 3.0, 2.0])
```

### `deadband_spline` — Smooth with guarantee

```python
from spline_midi_smooth.deadband_spline import deadband_spline

times, values, upper, lower = deadband_spline(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8), (3, -0.1)],
    epsilon=1.0,
    smooth_method="cubic_hermite",  # or "catmull_rom"
)
# Guarantee: lower[i] <= values[i] <= upper[i] for all i
```

If control points violate the deadband (any |y| > ε), raises `ValueError`.

Cubic splines can overshoot between control points. The function clamps any excursion back to the deadband (conservative guarantee).

### `deadband_spline_exact_proof` — Linear spline proof

```python
from spline_midi_smooth.deadband_spline import deadband_spline_exact_proof

proven = deadband_spline_exact_proof(
    phase_offsets=[(0, 0.5), (1, -0.3), (2, 0.8)],
    epsilon=1.0,
)
# True — the convexity argument holds: if |P0|≤ε and |P1|≤ε,
# then |(1-λ)P0 + λP1| ≤ ε for all λ∈[0,1]
```

### `is_deadband_a_spline` — Structured proof

```python
from spline_midi_smooth.deadband_spline import is_deadband_a_spline

proof = is_deadband_a_spline([0.0, 1.0, 2.0], epsilon=5.0)
print(proof["proposition"])  # "A deadband funnel is a piecewise-linear spline"
print(proof["conclusion"])   # "The deadband funnel is exactly a pair of degree-1 B-splines."
```

Returns a structured proof record with knot vectors, control points, and step-by-step evidence.

## Input/output formats

### Input

- **MIDI files**: Standard MIDI (Type 0 or Type 1). Parsed via `mido`.
- **Control points**: `Sequence[tuple[float, float]]` for direct spline construction. x must be strictly increasing.
- **CC values**: 0–127 (7-bit unsigned)
- **Pitch bend**: −8192 to +8191 (14-bit signed)
- **Velocity**: 1–127
- **Tempo**: microseconds per quarter note (via `set_tempo` meta events)

### Output

- **MIDI files**: Type 0 (single track). All events merged into one timeline.
- **Spline functions**: `callable(x) -> float | np.ndarray`
- **Deadband arrays**: `np.ndarray` of float64

## Configuration reference

### `smooth_midi_cc`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | required | Source MIDI file |
| `output_path` | `str \| Path` | required | Destination MIDI file |
| `method` | `str` | `"cubic_hermite"` | Spline kernel |
| `rate_hz` | `float` | `1000.0` | Output sampling rate |
| `keep_original_cc` | `bool` | `False` | Retain original events |

### `smooth_midi_volume`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | required | Source MIDI |
| `output_path` | `str \| Path` | required | Destination MIDI |
| `channel` | `int \| None` | `None` | Target channel |
| `method` | `str` | `"cubic_hermite"` | Spline kernel |
| `rate_hz` | `float` | `1000.0` | Sampling rate |

### `smooth_pitch_bend`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | required | Source MIDI |
| `output_path` | `str \| Path` | required | Destination MIDI |
| `method` | `str` | `"cubic_hermite"` | `"cubic_hermite"` or `"catmull_rom"` |
| `rate_hz` | `float` | `2000.0` | Sampling rate |
| `channel` | `int \| None` | `None` | Target channel |

### `smooth_velocity_curve`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | required | Source MIDI |
| `output_path` | `str \| Path` | required | Destination MIDI |
| `method` | `str` | `"catmull_rom"` | Spline kernel |
| `rate_hz` | `float` | `500.0` | Sampling rate |
| `channel` | `int \| None` | `None` | Target channel |

### `smooth_tempo_map`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | required | Source MIDI |
| `output_path` | `str \| Path` | required | Destination MIDI |
| `method` | `str` | `"cubic_hermite"` | Spline kernel |
| `rate_hz` | `float` | `10.0` | Sampling rate |

### `deadband_spline`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_offsets` | `Sequence[tuple]` | required | (time, offset) points |
| `epsilon` | `float` | required | Deadband half-width |
| `smooth_method` | `str` | `"cubic_hermite"` | Kernel |

## Use cases

### 1. Remove zipper noise from volume automation

```python
from spline_midi_smooth.midi_processor import smooth_midi_volume

smooth_midi_volume("raw.mid", "smooth.mid", channel=0, rate_hz=1000)
```

### 2. Smooth filter cutoff sweeps

```python
from spline_midi_smooth.midi_processor import smooth_midi_cc

stats = smooth_midi_cc("filter_sweep.mid", "smooth_sweep.mid", rate_hz=2000)
# CC#74 (filter cutoff) will be resampled at 2kHz
```

### 3. Create human-like velocity curves

```python
from spline_midi_smooth.anti_alias import smooth_velocity_curve

smooth_velocity_curve("drum_machine.mid", "human.mid", method="catmull_rom")
```

### 4. Smooth pitch bend for expressive solos

```python
from spline_midi_smooth.anti_alias import smooth_pitch_bend

smooth_pitch_bend("bend.mid", "smooth_bend.mid", rate_hz=2000)
```

### 5. Convert rigid tempo changes into natural rubato

```python
from spline_midi_smooth.anti_alias import smooth_tempo_map

smooth_tempo_map("rigid.mid", "rubato.mid", rate_hz=10)
```

### 6. Verify a control signal stays within tolerance

```python
from spline_midi_smooth.deadband_spline import deadband_spline_exact_proof

points = [(0, 0.1), (0.5, -0.2), (1.0, 0.05)]
assert deadband_spline_exact_proof(points, epsilon=0.5)
```

### 7. Build a custom spline for reverb send automation

```python
from spline_midi_smooth.interpolation import catmull_rom
import numpy as np

# Reverb wet/dry curve: gradual swell then decay
points = [(0, 0), (2, 40), (4, 90), (6, 80), (8, 30), (10, 0)]
spline = catmull_rom(points, tension=0.3)

# Sample at MIDI CC resolution
t = np.linspace(0, 10, 10000)
cc_values = np.clip(np.rint(spline(t)), 0, 127).astype(int)
```

## Troubleshooting

### "x coordinates must be strictly increasing"

All spline functions require monotonically increasing x values. Check for duplicate timestamps in your control points. If you have multiple events at the same time, you'll need to deduplicate or offset them slightly.

### "cubic_hermite requires at least 2 points"

All splines need a minimum of 2 points. B-spline needs at least `degree + 1` points (4 for degree 3).

### Output MIDI sounds the same

- Increase `rate_hz` — you may not be sampling densely enough
- Check that the input MIDI actually has CC automation (not just note events)
- Try `keep_original_cc=True` to verify the original events exist

### B-spline doesn't pass through my points

That's expected — B-spline is an *approximating* spline. Use `cubic_hermite` or `catmull_rom` if you need interpolation.

### Pitch bend values seem wrong after smoothing

Pitch bend is 14-bit signed (−8192 to +8191). The smoother clamps to this range. If your original values are near the extremes, the spline may overshoot and get clamped, causing flat spots at the boundaries.

### "control points violate deadband"

Your phase offsets have a value exceeding `epsilon`. Either increase `epsilon` or reduce the offset magnitudes.

### Large MIDI files are slow

The B-spline kernel uses recursive Cox-de Boor which is O(n × degree) per evaluation point. For large files, prefer `cubic_hermite` or `catmull_rom` which are O(n) per evaluation.
