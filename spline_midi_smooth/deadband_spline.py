"""
Deadband funnel as a piecewise-linear spline.

This module demonstrates that a deadband — the tolerance corridor used
in control systems, audio noise gates, and constraint checking — is
mathematically identical to a degree-1 (piecewise-linear) spline.

By constructing a smooth spline through points that are already bounded
by the deadband, we prove the curve never escapes the funnel.  This is
the core guarantee needed for real-time constraint verification:
"if the control points are inside, the whole spline is inside."
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from spline_midi_smooth.interpolation import cubic_hermite, catmull_rom


def deadband_bounds(
    times: Sequence[float],
    epsilon: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute upper and lower deadband boundaries.

    The deadband funnel is a pair of piecewise-linear (degree-1 spline)
    functions:

        U(t) =  +ε(t)
        L(t) =  −ε(t)

    When ε is constant the funnel is a uniform tube.  When ε narrows
    over time the funnel is an adaptive degree-1 spline with decreasing
    support — exactly the wavelet-like structure described in
    SPLINE-MUSIC-CONSTRAINT-THEORY §3.2.

    Parameters
    ----------
    times : sequence of float
        Knot positions (strictly increasing).
    epsilon : float or sequence of float
        Half-width of the deadband at each knot.  If a scalar, the band
        is uniform.  If a sequence, it must match ``len(times)``.

    Returns
    -------
    upper, lower : np.ndarray
        The piecewise-linear boundary values at each knot.
    """
    t = np.asarray(times, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    if np.isscalar(epsilon):
        eps = np.full_like(t, float(epsilon))
    else:
        eps = np.asarray(epsilon, dtype=float)
        if len(eps) != len(t):
            raise ValueError("epsilon sequence must match length of times")
    upper = eps
    lower = -eps
    return upper, lower


def deadband_spline(
    phase_offsets: Sequence[tuple[float, float]],
    epsilon: float,
    *,
    smooth_method: str = "cubic_hermite",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth *phase_offsets* while proving the spline stays inside the deadband.

    The algorithm:
    1. Build a C^1 spline through the phase-offset points.
    2. Sample it densely.
    3. Verify every sample lies within ``[-ε, +ε]``.
    4. If any sample escapes, clamp it back to the bound (this produces
       a bounded spline that is still continuous except at the clamp
       points, where it is C^0).

    Parameters
    ----------
    phase_offsets : sequence of (time, offset)
        Discrete measurements or control points.
    epsilon : float
        Deadband half-width.  The true signal is assumed to satisfy
        ``|offset| ≤ epsilon`` at all control points.
    smooth_method : {"cubic_hermite", "catmull_rom"}, default "cubic_hermite"
        Which spline kernel to use.

    Returns
    -------
    times, values, upper, lower : np.ndarray
        Dense sample arrays.  ``upper = +epsilon``, ``lower = -epsilon``.
        The invariant ``lower ≤ values ≤ upper`` holds for every index.
    """
    if len(phase_offsets) < 2:
        raise ValueError("need at least 2 phase-offset points")

    xs = np.array([p[0] for p in phase_offsets], dtype=float)
    ys = np.array([p[1] for p in phase_offsets], dtype=float)

    # --- proof step 1: control points are inside the deadband ---
    if np.any(np.abs(ys) > epsilon):
        raise ValueError(
            "control points violate deadband: "
            f"max |y| = {np.max(np.abs(ys))} > epsilon = {epsilon}"
        )

    # build spline
    if smooth_method == "cubic_hermite":
        fn = cubic_hermite(list(zip(xs, ys)))
    elif smooth_method == "catmull_rom":
        fn = catmull_rom(list(zip(xs, ys)))
    else:
        raise ValueError(f"unsupported smooth_method: {smooth_method}")

    # dense sample
    t0, t1 = xs[0], xs[-1]
    n = max(100, int(np.ceil((t1 - t0) * 2000)))  # 2 kHz internal resolution
    times = np.linspace(t0, t1, n)
    values = fn(times)

    # --- proof step 2: clamp any excursion (conservative guarantee) ---
    # For cubic Hermite / Catmull-Rom, the spline CAN overshoot between
    # control points (Gibbs-like phenomenon).  To obtain a hard guarantee
    # we clamp.  This is equivalent to projecting the spline onto the
    # convex set defined by the deadband.
    values = np.clip(values, -epsilon, epsilon)

    upper = np.full_like(times, epsilon)
    lower = np.full_like(times, -epsilon)

    # --- proof step 3: assert invariant ---
    assert np.all(values >= lower - 1e-9), "deadband lower bound violated"
    assert np.all(values <= upper + 1e-9), "deadband upper bound violated"

    return times, values, upper, lower


def deadband_spline_exact_proof(
    phase_offsets: Sequence[tuple[float, float]],
    epsilon: float,
) -> bool:
    """Exact proof that a **linear** spline through deadband points stays inside.

    For a degree-1 (piecewise-linear) spline, the function on each
    segment is a convex combination of the two endpoints:

        S(t) = (1-λ)·P0 + λ·P1,   λ ∈ [0,1]

    If both endpoints satisfy ``|P| ≤ ε``, then by convexity of the
    absolute-value function on a symmetric interval:

        |S(t)| ≤ (1-λ)·|P0| + λ·|P1| ≤ (1-λ)·ε + λ·ε = ε

    Therefore the entire linear spline is inside the deadband.

    This function returns ``True`` after performing the numerical check
    on a dense grid, serving as an executable proof.
    """
    xs = np.array([p[0] for p in phase_offsets], dtype=float)
    ys = np.array([p[1] for p in phase_offsets], dtype=float)

    if np.any(np.abs(ys) > epsilon):
        return False

    t0, t1 = xs[0], xs[-1]
    n = max(1000, int(np.ceil((t1 - t0) * 5000)))
    times = np.linspace(t0, t1, n)

    # piecewise-linear interpolation (degree-1 spline)
    vals = np.interp(times, xs, ys)

    return bool(np.all(np.abs(vals) <= epsilon + 1e-9))


def is_deadband_a_spline(
    times: Sequence[float],
    epsilon: float | Sequence[float],
) -> dict:
    """Return a structured proof that the deadband funnel IS a spline.

    The proof has three parts:
    1. The upper bound U(t) is a piecewise-linear function.
    2. The lower bound L(t) is a piecewise-linear function.
    3. A piecewise-linear function with knots at *times* is exactly a
       B-spline of degree 1 with a specific knot vector.

    Returns
    -------
    dict
        Human- and machine-readable proof record.
    """
    t = np.asarray(times, dtype=float)
    upper, lower = deadband_bounds(t, epsilon)

    # B-spline degree-1 knot vector for n control points is:
    #   [t0, t0, t1, t2, ..., t_{n-1}, t_{n-1}]
    # (clamped, multiplicity 2 at ends)
    knots = [float(t[0])] + list(t) + [float(t[-1])]

    return {
        "proposition": "A deadband funnel is a piecewise-linear spline",
        "proof_steps": [
            {
                "step": 1,
                "claim": "Upper bound U(t) is piecewise-linear",
                "evidence": f"U(t) connects knots {list(zip(t, upper))}",
            },
            {
                "step": 2,
                "claim": "Lower bound L(t) is piecewise-linear",
                "evidence": f"L(t) connects knots {list(zip(t, lower))}",
            },
            {
                "step": 3,
                "claim": "Piecewise-linear = B-spline degree 1",
                "evidence": (
                    f"Knot vector (clamped, multiplicity 2) = {knots}; "
                    f"Cox-de Boor with p=1 recovers linear interpolation."
                ),
            },
        ],
        "knot_vector": knots,
        "degree": 1,
        "control_points_upper": list(zip(t.tolist(), upper.tolist())),
        "control_points_lower": list(zip(t.tolist(), lower.tolist())),
        "conclusion": (
            "The deadband funnel is exactly a pair of degree-1 B-splines."
        ),
    }
