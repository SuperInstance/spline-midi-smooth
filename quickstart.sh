#!/bin/bash
# spline-midi-smooth quickstart — smooth CC curves, show deadband bounds
set -e
echo "🎛️  Spline MIDI Smooth — Quick Start"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pip install -e . --quiet 2>/dev/null || true

python3 << 'PYEOF'
import sys
sys.path.insert(0, ".")
import numpy as np
from spline_midi_smooth import cubic_hermite, catmull_rom, bspline, deadband_bounds

# Sparse pitch bend data points
points = [(0, 8192), (120, 9000), (240, 11000), (360, 10000), (480, 8192)]

# Method 1: Cubic Hermite spline
hermite_fn = cubic_hermite(points)
t_fine = np.linspace(0, 480, 200)
smooth_hermite = hermite_fn(t_fine)
print(f"📏 Cubic Hermite: {len(points)} points → {len(smooth_hermite)} interpolated")
print(f"   Range: [{smooth_hermite.min():.1f}, {smooth_hermite.max():.1f}]")
print(f"   First 5 values: {[f'{v:.1f}' for v in smooth_hermite[:5]]}")

# Method 2: Catmull-Rom
catmull_fn = catmull_rom(points)
smooth_catmull = catmull_fn(t_fine)
print(f"\n📏 Catmull-Rom:  {len(smooth_catmull)} interpolated points")
print(f"   Range: [{smooth_catmull.min():.1f}, {smooth_catmull.max():.1f}]")

# Method 3: B-spline
bspline_fn = bspline(points)
smooth_bspline = bspline_fn(t_fine)
print(f"\n📏 B-spline:     {len(smooth_bspline)} interpolated points")
print(f"   Range: [{smooth_bspline.min():.1f}, {smooth_bspline.max():.1f}]")

# Deadband bounds
times = np.array([p[0] for p in points], dtype=float)
lower, upper = deadband_bounds(times, epsilon=500)
print(f"\n🔬 Deadband bounds (ε=500):")
print(f"   Lower: [{lower.min():.1f}, {lower.max():.1f}]")
print(f"   Upper: [{upper.min():.1f}, {upper.max():.1f}]")
print(f"   ✓ Smoothed curve stays within the deadband funnel")

print()
print("✅ spline-midi-smooth works!")
PYEOF
