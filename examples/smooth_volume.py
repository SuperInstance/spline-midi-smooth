"""Demo: Smooth MIDI CC volume data to remove zipper noise."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_midi_smooth import smooth_midi_volume, catmull_rom

# Sparse volume CC events (controller 7 = volume)
volume_events = [
    (0, 80),    # Start at 80
    (96, 80),   # Hold
    (192, 100), # Swell up
    (288, 120), # Peak
    (384, 90),  # Dip
    (480, 80),  # Return
]

# Smooth with 4x density — eliminates zipper noise
smoothed = smooth_midi_volume(volume_events, density=4)
print(f"Volume CC: {len(volume_events)} events → {len(smoothed)} interpolated")
for tick, val in smoothed[::20]:  # Print every 20th point
    bar = "█" * int(val / 8)
    print(f"  tick {tick:4d}: {val:6.1f}  {bar}")

# Verify smoothness: max step between adjacent points
max_step = max(abs(smoothed[i+1][1] - smoothed[i][1]) for i in range(len(smoothed)-1))
print(f"\nMax step between adjacent values: {max_step:.2f}")
print("  (< 1.0 = no audible zipper noise)")
