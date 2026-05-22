"""
Spline interpolation for MIDI CC automation.

Provides continuous curves from discrete MIDI control-change points,
bridging the discrete→continuous gap that causes "zipper noise" in
digital parameter changes.

Mathematical foundations drawn from SPLINE-MUSIC-CONSTRAINT-THEORY:
- B-Splines  :: numerically stable basis-spline representation
- Hermite    :: local C^1 interpolant with explicit tangents
- Catmull-Rom:: interpolating spline with automatic tangent estimation
"""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np


def _finite_difference_tangents(
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Estimate tangents at each point using central differences.

    Endpoints use one-sided differences so the spline can still be
    evaluated across the full domain.
    """
    n = len(xs)
    ts = np.empty_like(ys, dtype=float)
    # Forward difference at start
    ts[0] = (ys[1] - ys[0]) / (xs[1] - xs[0])
    # Central differences interior
    for i in range(1, n - 1):
        dx_left = xs[i] - xs[i - 1]
        dx_right = xs[i + 1] - xs[i]
        # weighted average to handle non-uniform spacing
        ts[i] = 0.5 * ((ys[i] - ys[i - 1]) / dx_left + (ys[i + 1] - ys[i]) / dx_right)
    # Backward difference at end
    ts[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return ts


def _hermite_basis(t: float) -> tuple[float, float, float, float]:
    """Cubic Hermite basis functions at parameter t in [0, 1].

    Returns (h00, h10, h01, h11) where
        H(t) = h00*P0 + h10*T0 + h01*P1 + h11*T1
    """
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00, h10, h01, h11


def cubic_hermite(
    points: Sequence[tuple[float, float]],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Build a C^1 cubic Hermite spline through *points*.

    Tangents are estimated automatically via finite differences so the
    caller does not need to supply derivatives.

    Parameters
    ----------
    points : sequence of (x, y)
        At least two points, strictly increasing in *x*.

    Returns
    -------
    spline_fn : callable
        ``spline_fn(x)`` evaluates the spline at *x* (scalar or array).
        Values outside the domain are clamped to the nearest endpoint.
    """
    if len(points) < 2:
        raise ValueError("cubic_hermite requires at least 2 points")
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("x coordinates must be strictly increasing")
    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
        raise ValueError("Input contains NaN or Inf values")

    ts = _finite_difference_tangents(xs, ys)

    def _eval(x: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(x).astype(float)
        out = np.empty_like(x_arr, dtype=float)

        for idx, xv in np.ndenumerate(x_arr):
            # clamp to domain
            if xv <= xs[0]:
                out[idx] = ys[0]
                continue
            if xv >= xs[-1]:
                out[idx] = ys[-1]
                continue
            # locate segment
            seg = int(np.searchsorted(xs, xv, side="right") - 1)
            seg = max(0, min(seg, len(xs) - 2))
            # local parameter t in [0, 1]
            dx = xs[seg + 1] - xs[seg]
            t = (xv - xs[seg]) / dx
            h00, h10, h01, h11 = _hermite_basis(t)
            # scale tangents by segment width for parametric invariance
            out[idx] = (
                h00 * ys[seg]
                + h10 * dx * ts[seg]
                + h01 * ys[seg + 1]
                + h11 * dx * ts[seg + 1]
            )

        return float(out[0]) if scalar else out

    return _eval


def catmull_rom(
    points: Sequence[tuple[float, float]],
    tension: float = 0.5,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Build a Catmull-Rom spline through *points*.

    The Catmull-Rom spline is a special case of the cubic Hermite spline
    where tangents are derived from neighbouring control points:

        T_i = tension * (P_{i+1} - P_{i-1})

    With the standard tension factor 0.5 this yields a C^1 curve that
    interpolates every control point.

    Parameters
    ----------
    points : sequence of (x, y)
        At least two points, strictly increasing in *x*.
    tension : float, default 0.5
        Controls how "tight" the curve is.  0.0 is linear interpolation,
        0.5 is the classic Catmull-Rom, 1.0 is quite loose.

    Returns
    -------
    spline_fn : callable
        ``spline_fn(x)`` evaluates the spline at *x*.
    """
    if len(points) < 2:
        raise ValueError("catmull_rom requires at least 2 points")
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("x coordinates must be strictly increasing")
    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
        raise ValueError("Input contains NaN or Inf values")

    n = len(xs)
    ts = np.empty_like(ys, dtype=float)
    # Endpoint handling: duplicate first/last point (phantom knots)
    # so the curve still starts/ends at the first/last real point.
    for i in range(n):
        if i == 0:
            ts[i] = tension * (ys[1] - ys[0])
        elif i == n - 1:
            ts[i] = tension * (ys[-1] - ys[-2])
        else:
            ts[i] = tension * (ys[i + 1] - ys[i - 1])

    def _eval(x: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(x).astype(float)
        out = np.empty_like(x_arr, dtype=float)

        for idx, xv in np.ndenumerate(x_arr):
            if xv <= xs[0]:
                out[idx] = ys[0]
                continue
            if xv >= xs[-1]:
                out[idx] = ys[-1]
                continue
            seg = int(np.searchsorted(xs, xv, side="right") - 1)
            seg = max(0, min(seg, n - 2))
            dx = xs[seg + 1] - xs[seg]
            t = (xv - xs[seg]) / dx
            h00, h10, h01, h11 = _hermite_basis(t)
            out[idx] = (
                h00 * ys[seg]
                + h10 * dx * ts[seg]
                + h01 * ys[seg + 1]
                + h11 * dx * ts[seg + 1]
            )

        return float(out[0]) if scalar else out

    return _eval


def _cox_de_boor(
    u: float,
    knots: np.ndarray,
    i: int,
    p: int,
) -> float:
    """Recursive Cox-de Boor formula for B-spline basis N_{i,p}(u).

    Pure Python version; fine for low-degree splines and modest
    sample counts.  For heavy real-time use this would be vectorised.
    """
    if p == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        # special case: last knot inclusive
        if abs(u - knots[-1]) < 1e-12 and i == len(knots) - 2:
            return 1.0
        return 0.0

    left_denom = knots[i + p] - knots[i]
    right_denom = knots[i + p + 1] - knots[i + 1]

    left = 0.0
    right = 0.0
    if left_denom > 1e-12:
        left = (u - knots[i]) / left_denom * _cox_de_boor(u, knots, i, p - 1)
    if right_denom > 1e-12:
        right = (knots[i + p + 1] - u) / right_denom * _cox_de_boor(
            u, knots, i + 1, p - 1
        )
    return left + right


def bspline(
    points: Sequence[tuple[float, float]],
    degree: int = 3,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Build a uniform clamped B-spline curve from *points*.

    The curve is defined by B-spline basis functions (Cox-de Boor) and
    does **not** in general pass through the control points (it is an
    approximating spline).  The clamped knot vector forces interpolation
    at the first and last control points.

    Parameters
    ----------
    points : sequence of (x, y)
        Control points.  Number must be >= degree + 1.
    degree : int, default 3
        Polynomial degree of each segment (cubic when degree=3).

    Returns
    -------
    spline_fn : callable
        ``spline_fn(x)`` evaluates the spline at *x*.
        The input domain is mapped from the original x-range to the
        parametric domain [0, 1].
    """
    if len(points) < degree + 1:
        raise ValueError(
            f"bspline needs at least {degree + 1} points for degree {degree}"
        )

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
        raise ValueError("Input contains NaN or Inf values")
    n = len(xs) - 1  # control point index 0..n

    # Normalise parameter to [0, 1] based on original x span
    x_min, x_max = xs[0], xs[-1]
    x_range = x_max - x_min

    # Clamped knot vector: degree+1 zeros, 1..n-degree, degree+1 ones
    internal_knots = np.linspace(0.0, 1.0, n - degree + 2)
    knots = np.concatenate([
        np.full(degree, 0.0),
        internal_knots,
        np.full(degree, 1.0),
    ])

    def _eval(x: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(x).astype(float)
        out = np.empty_like(x_arr, dtype=float)

        for idx, xv in np.ndenumerate(x_arr):
            # clamp to domain
            if xv <= x_min:
                out[idx] = ys[0]
                continue
            if xv >= x_max:
                out[idx] = ys[-1]
                continue

            u = (xv - x_min) / x_range if x_range > 0 else 0.0
            val = 0.0
            for i in range(n + 1):
                val += _cox_de_boor(u, knots, i, degree) * ys[i]
            out[idx] = val

        return float(out[0]) if scalar else out

    return _eval
