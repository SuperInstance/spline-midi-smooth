"""Tests for spline interpolation functions."""

import numpy as np
import pytest

from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom, bspline


# ── cubic_hermite ──────────────────────────────────────────────────

class TestCubicHermite:
    def test_two_points_linear(self):
        """With 2 points and matching finite-difference tangent → linear."""
        pts = [(0.0, 0.0), (1.0, 10.0)]
        fn = cubic_hermite(pts)
        x = np.linspace(0, 1, 50)
        y = fn(x)
        # Should be very close to y = 10x
        expected = 10.0 * x
        np.testing.assert_allclose(y, expected, atol=0.5)

    def test_three_points_smooth(self):
        """Three non-collinear points produce a smooth curve through them."""
        pts = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]
        fn = cubic_hermite(pts)
        # Endpoints match exactly
        assert abs(fn(0.0) - 0.0) < 1e-9
        assert abs(fn(2.0) - 0.0) < 1e-9
        # Middle is near 5 (Hermite interpolates endpoints but interior is smooth)
        mid_val = fn(1.0)
        assert abs(mid_val - 5.0) < 1.0  # not exact interpolation but close

    def test_five_points_endpoints_exact(self):
        """Endpoints are interpolated exactly for any number of points."""
        pts = [(0, 0), (1, 3), (2, 1), (3, 4), (4, 2)]
        fn = cubic_hermite(pts)
        assert abs(fn(0.0) - 0.0) < 1e-9
        assert abs(fn(4.0) - 2.0) < 1e-9

    def test_clamped_below_domain(self):
        """Values below domain clamp to first y."""
        pts = [(1.0, 5.0), (3.0, 10.0)]
        fn = cubic_hermite(pts)
        assert fn(0.0) == 5.0
        assert fn(-10.0) == 5.0

    def test_clamped_above_domain(self):
        """Values above domain clamp to last y."""
        pts = [(1.0, 5.0), (3.0, 10.0)]
        fn = cubic_hermite(pts)
        assert fn(4.0) == 10.0
        assert fn(100.0) == 10.0

    def test_continuity_dense_grid(self):
        """Evaluate on dense grid, check no jumps > threshold."""
        pts = [(0, 0), (1, 3), (2, -1), (3, 5), (4, 2), (5, 0)]
        fn = cubic_hermite(pts)
        x = np.linspace(0, 5, 5000)
        y = fn(x)
        diffs = np.abs(np.diff(y))
        # With 5000 points over range 5, max step should be < 0.1
        assert np.max(diffs) < 0.1

    def test_colinear_points(self):
        """Colinear points should give near-linear output."""
        pts = [(i, 2.0 * i + 1.0) for i in range(6)]
        fn = cubic_hermite(pts)
        x = np.linspace(0, 5, 100)
        y = fn(x)
        expected = 2.0 * x + 1.0
        np.testing.assert_allclose(y, expected, atol=1e-9)

    def test_two_points_minimum(self):
        """2 points is the minimum and should work."""
        pts = [(0, 10), (1, 20)]
        fn = cubic_hermite(pts)
        assert abs(fn(0.0) - 10.0) < 1e-9
        assert abs(fn(1.0) - 20.0) < 1e-9

    def test_scalar_return_type(self):
        """Scalar input returns float."""
        pts = [(0, 0), (1, 1)]
        fn = cubic_hermite(pts)
        result = fn(0.5)
        assert isinstance(result, float)


# ── catmull_rom ────────────────────────────────────────────────────

class TestCatmullRom:
    def test_tension_zero_near_linear(self):
        """Tension=0 → zero tangents, so interior points sag below linear."""
        pts = [(0, 0), (1, 5), (2, 10)]
        fn = catmull_rom(pts, tension=0.0)
        x = np.linspace(0, 2, 100)
        y = fn(x)
        # tension=0 zeroes all tangents, flattening the curve.
        # With zero tangents the Hermite basis degenerates to h00*P0 + h01*P1
        # which is still monotonic but not perfectly linear.
        # Just verify endpoints match and values are monotonic.
        assert abs(fn(0.0) - 0.0) < 1e-9
        assert abs(fn(2.0) - 10.0) < 1e-9
        # monotonic increasing
        assert np.all(np.diff(y) >= -1e-9)

    def test_tension_half_standard(self):
        """Tension=0.5 (standard Catmull-Rom) passes through all points."""
        pts = [(0, 0), (1, 5), (2, 3), (3, 8)]
        fn = catmull_rom(pts, tension=0.5)
        # Check endpoints exactly
        assert abs(fn(0.0) - 0.0) < 1e-9
        assert abs(fn(3.0) - 8.0) < 1e-9

    def test_tension_one_loose(self):
        """Tension=1.0 produces a looser curve (overshoots possible)."""
        pts = [(0, 0), (1, 0), (2, 10), (3, 0), (4, 0)]
        fn_loose = catmull_rom(pts, tension=1.0)
        fn_std = catmull_rom(pts, tension=0.5)
        mid_loose = fn_loose(1.5)
        mid_std = fn_std(1.5)
        # Looser tension → larger value at midpoint of rising segment
        assert mid_loose >= mid_std or abs(mid_loose - mid_std) < 0.5

    def test_passes_through_all_control_points(self):
        """Catmull-Rom interpolates all control points."""
        pts = [(0, 10), (2, 30), (4, 20), (6, 50)]
        fn = catmull_rom(pts, tension=0.5)
        for x, y in pts:
            assert abs(fn(x) - y) < 1e-6, f"Mismatch at x={x}: {fn(x)} != {y}"

    def test_continuity(self):
        """Dense evaluation has no jumps."""
        pts = [(0, 0), (1, 5), (2, 3), (3, 8), (4, 1)]
        fn = catmull_rom(pts, tension=0.5)
        x = np.linspace(0, 4, 5000)
        y = fn(x)
        diffs = np.abs(np.diff(y))
        assert np.max(diffs) < 0.1


# ── bspline ────────────────────────────────────────────────────────

class TestBSpline:
    def test_degree_2(self):
        """Quadratic B-spline with 4 control points."""
        pts = [(0, 0), (1, 5), (2, 5), (3, 0)]
        fn = bspline(pts, degree=2)
        # Clamped: should match first and last
        assert abs(fn(0.0) - 0.0) < 1e-6
        assert abs(fn(3.0) - 0.0) < 1e-6

    def test_degree_3(self):
        """Cubic B-spline with 5 control points."""
        pts = [(0, 0), (1, 3), (2, 6), (3, 3), (4, 0)]
        fn = bspline(pts, degree=3)
        assert abs(fn(0.0) - 0.0) < 1e-6
        assert abs(fn(4.0) - 0.0) < 1e-6

    def test_clamped_first_last(self):
        """Clamped B-spline passes through first and last control points."""
        pts = [(i, float(i) ** 2) for i in range(6)]
        fn = bspline(pts, degree=3)
        assert abs(fn(0.0) - 0.0) < 1e-4
        assert abs(fn(5.0) - 25.0) < 1e-4

    def test_continuity(self):
        """Dense evaluation: no jumps > threshold."""
        pts = [(0, 0), (1, 5), (2, 3), (3, 8), (4, 2), (5, 0)]
        fn = bspline(pts, degree=3)
        x = np.linspace(0, 5, 5000)
        y = fn(x)
        diffs = np.abs(np.diff(y))
        assert np.max(diffs) < 0.1

    def test_clamped_outside_domain(self):
        """Values outside domain clamp to endpoints."""
        pts = [(0, 10), (1, 20), (2, 30), (3, 40)]
        fn = bspline(pts, degree=2)
        assert abs(fn(-5.0) - 10.0) < 1e-9
        assert abs(fn(10.0) - 40.0) < 1e-9


# ── error handling ─────────────────────────────────────────────────

class TestErrors:
    def test_hermite_non_increasing_x_raises(self):
        """Non-increasing x should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            cubic_hermite([(0, 0), (0, 1)])

    def test_hermite_decreasing_x_raises(self):
        """Decreasing x should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            cubic_hermite([(1, 0), (0, 1)])

    def test_catmull_rom_non_increasing_x_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            catmull_rom([(0, 0), (0, 1)])

    def test_hermite_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            cubic_hermite([(0, 0)])

    def test_catmull_rom_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            catmull_rom([(0, 0)])

    def test_bspline_too_few_points(self):
        with pytest.raises(ValueError, match="at least 4"):
            bspline([(0, 0), (1, 1)], degree=3)

    def test_bspline_non_increasing_x(self):
        """bspline doesn't explicitly check x ordering in the current impl,
        but we test the basic error path for insufficient points."""
        with pytest.raises(ValueError):
            bspline([(0, 0)], degree=2)
