"""Tests for deadband_spline module."""

import numpy as np
import pytest

from spline_midi_smooth.deadband_spline import (
    deadband_bounds,
    deadband_spline,
    deadband_spline_exact_proof,
    is_deadband_a_spline,
)


class TestDeadbandBounds:
    def test_uniform_epsilon(self):
        upper, lower = deadband_bounds([0, 1, 2], epsilon=0.1)
        np.testing.assert_allclose(upper, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(lower, [-0.1, -0.1, -0.1])

    def test_varying_epsilon(self):
        upper, lower = deadband_bounds([0, 1, 2], epsilon=[0.1, 0.5, 0.2])
        np.testing.assert_allclose(upper, [0.1, 0.5, 0.2])
        np.testing.assert_allclose(lower, [-0.1, -0.5, -0.2])

    def test_non_increasing_times_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            deadband_bounds([1, 1, 2], epsilon=0.1)


class TestDeadbandSpline:
    def test_output_within_epsilon_0_01(self):
        pts = [(0, 0.005), (0.5, -0.008), (1.0, 0.003)]
        times, values, upper, lower = deadband_spline(pts, epsilon=0.01)
        assert np.all(values >= lower - 1e-9)
        assert np.all(values <= upper + 1e-9)

    def test_output_within_epsilon_0_1(self):
        pts = [(0, 0.05), (1, -0.08), (2, 0.07), (3, 0.03)]
        times, values, upper, lower = deadband_spline(pts, epsilon=0.1)
        assert np.all(values >= -0.1 - 1e-9)
        assert np.all(values <= 0.1 + 1e-9)

    def test_output_within_epsilon_1_0(self):
        pts = [(0, 0.5), (1, -0.9), (2, 0.2)]
        times, values, upper, lower = deadband_spline(pts, epsilon=1.0)
        assert np.all(values >= -1.0 - 1e-9)
        assert np.all(values <= 1.0 + 1e-9)

    def test_violating_control_points_raise(self):
        with pytest.raises(ValueError, match="violate deadband"):
            deadband_spline([(0, 0.5), (1, 0.3)], epsilon=0.1)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            deadband_spline([(0, 0.0)], epsilon=0.1)

    def test_epsilon_zero_exact(self):
        """epsilon=0: all offsets must be 0.0."""
        pts = [(0, 0.0), (1, 0.0), (2, 0.0)]
        times, values, upper, lower = deadband_spline(pts, epsilon=0.0)
        np.testing.assert_allclose(values, 0.0, atol=1e-9)

    def test_catmull_rom_method(self):
        pts = [(0, 0.05), (1, -0.03), (2, 0.04)]
        times, values, upper, lower = deadband_spline(
            pts, epsilon=0.1, smooth_method="catmull_rom"
        )
        assert np.all(values >= -0.1 - 1e-9)
        assert np.all(values <= 0.1 + 1e-9)


class TestDeadbandSplineExactProof:
    def test_linear_inside_returns_true(self):
        pts = [(0, 0.05), (1, -0.03), (2, 0.04)]
        assert deadband_spline_exact_proof(pts, epsilon=0.1) is True

    def test_linear_violating_returns_false(self):
        pts = [(0, 0.5), (1, 0.3)]
        assert deadband_spline_exact_proof(pts, epsilon=0.1) is False


class TestIsDeadbandASpline:
    def test_returns_proof_dict(self):
        result = is_deadband_a_spline([0, 1, 2], epsilon=0.1)
        assert "proposition" in result
        assert result["degree"] == 1
        assert len(result["knot_vector"]) > 0
        assert len(result["proof_steps"]) == 3
