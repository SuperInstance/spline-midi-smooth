# Spline MIDI Smooth — Developer Guide

## Architecture

```
spline_midi_smooth/
├── interpolation.py    # Spline kernels (Hermite, Catmull-Rom, B-spline)
├── midi_processor.py   # MIDI CC smoothing pipeline
├── anti_alias.py       # Domain-specific: pitch bend, velocity, tempo
├── deadband_spline.py  # Deadband proofs and bounded spline guarantees
```

### Module Diagram

```
┌──────────────────┐
│ interpolation.py │  Core spline math: cubic_hermite, catmull_rom, bspline
└────────┬─────────┘
         │ used by
    ┌────┴──────────────┬─────────────────┐
    │                   │                 │
┌───▼──────────┐ ┌──────▼─────────┐ ┌─────▼──────────┐
│midi_processor│ │  anti_alias.py │ │deadband_spline │
│   .py        │ │ (pitch bend,   │ │  .py           │
│ (CC smooth)  │ │  velocity,     │ │ (proofs,       │
└──────────────┘ │  tempo)        │ │  guarantees)   │
                 └────────────────┘ └────────────────┘
```

### Design Decisions

- **Callable splines:** All kernels return `Callable[[float | ndarray], float | ndarray]` — evaluate at any point.
- **Clamping, not extrapolation:** Values outside the domain return the nearest endpoint value.
- **Cox-de Boor recursion:** B-spline basis computed via pure Python recursion. Fine for MIDI resolutions (hundreds to thousands of points).
- **Deadband = spline:** The `deadband_spline.py` module proves the deadband funnel is isomorphic to a degree-1 B-spline with specific knot vectors.

## Extending

### Adding a New Spline Kernel

Follow the pattern in `interpolation.py`:

```python
def monotone_cubic(
    points: Sequence[tuple[float, float]],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Fritsch-Carlson monotone cubic interpolation.

    Guarantees monotonicity between control points.
    """
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    # Compute monotone tangents (Fritsch-Carlson method)
    ...
    ts = _compute_monotone_tangents(xs, ys)

    def _eval(x):
        # Same evaluation loop as cubic_hermite
        ...
    return _eval
```

Then add it to the dispatch in `midi_processor.py` and `anti_alias.py`:

```python
if method == "monotone_cubic":
    spline_fn = monotone_cubic(pts)
```

### Adding a New MIDI Smoothing Target

Follow the pattern in `anti_alias.py`:

```python
def smooth_aftertouch(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "cubic_hermite",
    rate_hz: float = 200.0,
    channel: int | None = None,
) -> int:
    """Spline-interpolate channel aftertouch events."""
    mid_in = mido.MidiFile(str(input_path))
    events = _midi_to_absolute_seconds(mid_in)

    # Collect aftertouch events
    pressures = {}
    for sec, msg in events:
        if msg.type == "aftertouch":
            ...

    # Build spline, resample, rebuild MIDI file
    ...
    return n_generated
```

### Adding a New Deadband Proof

Extend `deadband_spline.py`:

```python
def deadband_cubic_proof(
    phase_offsets: Sequence[tuple[float, float]],
    epsilon: float,
) -> bool:
    """Prove a cubic spline stays inside the deadband using interval arithmetic."""
    # For each segment, compute the cubic's extrema analytically
    # and verify they lie within ±ε
    ...
```

## Testing

```bash
pytest                    # all tests
pytest -v                 # verbose
pytest --cov=spline_midi_smooth  # coverage
```

### Test Patterns

```python
def test_spline_interpolation():
    """Spline passes through control points."""
    points = [(0, 64), (1, 100), (2, 72)]
    fn = cubic_hermite(points)
    assert abs(fn(0.0) - 64.0) < 1e-9
    assert abs(fn(1.0) - 100.0) < 1e-9
    assert abs(fn(2.0) - 72.0) < 1e-9

def test_deadband_guarantee():
    """Values stay within ±ε."""
    offsets = [(0, 3), (1, -5), (2, 8), (3, -2)]
    t, v, upper, lower = deadband_spline(offsets, epsilon=10.0)
    assert np.all(v >= lower - 1e-9)
    assert np.all(v <= upper + 1e-9)
```

## Contributing

1. Fork, branch, implement, test, PR
2. Spline kernels go in `interpolation.py`
3. MIDI-domain smoothing goes in `anti_alias.py` or `midi_processor.py`
4. Mathematical proofs go in `deadband_spline.py`
5. All new functions need tests

### Code Style

- Python 3.10+ with type hints
- `numpy` for numerical arrays, `mido` for MIDI I/O
- Callable return pattern for splines
- Docstrings with Parameters/Returns

### Build System

```bash
pip install -e .
```

Dependencies: `mido>=1.3`, `numpy>=1.24`.
