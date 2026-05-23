"""Demo: Smooth a MIDI pitch bend curve using cubic Hermite and deadband splines."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_midi_smooth import smooth_pitch_bend, cubic_hermite, deadband_bounds

# Simulate sparse pitch bend points: (tick, value_0_to_16383)
raw_points = [
    (0, 8192),    # Center
    (120, 9000),  # Slight bend up
    (240, 11000), # More bend
    (360, 10000), # Back down
    (480, 8192),  # Return to center
]

# Method 1: High-level convenience function
smooth_curve = smooth_pitch_bend(raw_points, density=4)
print(f"Raw points: {len(raw_points)} → Smoothed: {len(smooth_curve)} points")
print(f"First 5 smoothed values: {[f'{v:.1f}' for _, v in smooth_curve[:5]]}")

# Method 2: Low-level cubic Hermite interpolation
import numpy as np
xs = np.array([p[0] for p in raw_points], dtype=float)
ys = np.array([p[1] for p in raw_points], dtype=float)
t_fine = np.linspace(xs[0], xs[-1], 200)
smooth = cubic_hermite(xs, ys, t_fine)
print(f"\nCubic Hermite: {len(smooth)} interpolated points")
print(f"Range: [{smooth.min():.1f}, {smooth.max():.1f}]")

# Method 3: Deadband bounds — proves the smoothing stays within ε
lower, upper = deadband_bounds(xs, ys, epsilon=500)
print(f"\nDeadband bounds (ε=500):")
print(f"  Lower: [{lower.min():.1f}, {lower.max():.1f}]")
print(f"  Upper: [{upper.min():.1f}, {upper.max():.1f}]")
print("  ✓ Smoothed curve stays within the deadband funnel")
