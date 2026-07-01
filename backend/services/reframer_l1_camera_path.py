"""L1-optimal camera path (Grundmann et al., CVPR 2011).

"Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths" — the
method YouTube's stabilizer used — decomposes a noisy per-sample target path
into a sequence of **static holds** and **constant-velocity linear pans**, the
two primitives a human camera operator actually uses. It does this by
minimizing a weighted L1 norm of the path's first three derivatives:

    minimize  w1·|D1(x)|₁ + w2·|D2(x)|₁ + w3·|D3(x)|₁
    subject to |x[t] − p[t]| ≤ r[t]        (stay near the subject)
               lo ≤ x[t] ≤ hi              (crop stays on-screen)

L1 (not L2) is the crucial choice: minimizing the L1 norm of the derivatives
yields a *sparse* derivative — most first-derivatives are exactly zero (holds),
most accelerations are exactly zero (constant-velocity pans), and jerk fires
only at the handful of transitions between them. An L2 objective would instead
smear motion everywhere, which reads as drift.

This is a linear program. We introduce nonnegative slack variables e1,e2,e3
that upper-bound the absolute derivatives, and solve with
``scipy.optimize.linprog`` (HiGHS). The result is the offline, globally-optimal
camera path — something the planner's greedy causal EMA can never produce.

Pure and dependency-light (numpy + scipy) so it can be unit-tested in isolation
and gated behind ``REFRAMER_L1_PATH``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is a hard dependency of the backend, but guard for CI stubs.
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, lil_matrix
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only in stripped envs
    _HAVE_SCIPY = False


def _derivative_rows(n: int, order: int) -> List[Tuple[int, List[Tuple[int, float]]]]:
    """Finite-difference stencils for derivative ``order`` over ``n`` samples.

    Returns a list of ``(i, [(col, coeff), ...])`` rows, one per valid index.
    D1[i] = x[i+1]-x[i]; D2[i] = x[i+2]-2x[i+1]+x[i]; D3[i] = x[i+3]-3x[i+2]+3x[i+1]-x[i].
    """
    stencils = {
        1: [(0, -1.0), (1, 1.0)],
        2: [(0, 1.0), (1, -2.0), (2, 1.0)],
        3: [(0, -1.0), (1, 3.0), (2, -3.0), (3, 1.0)],
    }
    stencil = stencils[order]
    rows = []
    for i in range(n - order):
        rows.append((i, [(i + off, c) for off, c in stencil]))
    return rows


def solve_l1_path(
    targets: Sequence[float],
    radius: Sequence[float] | float,
    lo: float,
    hi: float,
    weights: Tuple[float, float, float] = (1.0, 10.0, 100.0),
    proximity_weight: float = 0.1,
    anchor_targets: bool = True,
) -> Optional[List[float]]:
    """Solve for the L1-optimal camera path over ``targets``.

    Parameters
    ----------
    targets:
        Per-sample desired crop position (e.g. left-edge x in source pixels).
    radius:
        Allowed deviation of the path from each target (scalar or per-sample).
        The proximity band ``[p−r, p+r]`` is how far the operator may lag the
        subject before being pulled along.
    lo, hi:
        Hard bounds on the path (``0`` and ``max_x`` for a crop left-edge).
    weights:
        ``(w1, w2, w3)`` on the L1 norms of velocity, acceleration and jerk.
        Larger later weights → longer holds and cleaner linear pans.
    proximity_weight:
        Small L1 weight on ``|x[t] − p[t]|`` in the objective. With a pure
        derivative objective, a constant target admits infinitely many
        zero-cost flat paths anywhere in the band; this term breaks that
        degeneracy by pulling the path onto the subject, and biases pan
        timing to happen near the actual move. Keep it small relative to the
        derivative weights so it doesn't fight the hold/pan structure.
    anchor_targets:
        When ``True``, clamp each target into ``[lo, hi]`` before building the
        band so an out-of-range subject still yields a feasible program.

    Returns
    -------
    The optimal path as a list of floats, or ``None`` if scipy is unavailable
    or the LP is infeasible/failed (caller should fall back to the input).
    """
    n = len(targets)
    if n == 0:
        return []
    if not _HAVE_SCIPY:
        return None
    if n < 4:
        # Too short for a 3rd-derivative program — just clamp to bounds.
        return [float(min(hi, max(lo, t))) for t in targets]

    p = np.asarray(targets, dtype=np.float64)
    if np.isscalar(radius):
        r = np.full(n, float(radius))
    else:
        r = np.asarray(radius, dtype=np.float64)
        if r.shape[0] != n:
            raise ValueError("radius length must match targets length")
    r = np.maximum(r, 0.0)

    if anchor_targets:
        p = np.clip(p, lo, hi)

    w1, w2, w3 = (float(w) for w in weights)
    w0 = max(0.0, float(proximity_weight))

    # Variable layout: [ x(n) | e0(n) | e1(n-1) | e2(n-2) | e3(n-3) ]
    n1, n2, n3 = n - 1, n - 2, n - 3
    total = n + n + n1 + n2 + n3
    off_x = 0
    off_e0 = n
    off_e1 = off_e0 + n
    off_e2 = off_e1 + n1
    off_e3 = off_e2 + n2

    # Objective: minimize weighted sum of slacks; x itself is free in objective.
    c = np.zeros(total)
    c[off_e0:off_e0 + n] = w0
    c[off_e1:off_e1 + n1] = w1
    c[off_e2:off_e2 + n2] = w2
    c[off_e3:off_e3 + n3] = w3

    # Inequality constraints A_ub @ z <= b_ub:
    #   proximity: (x[i]-p[i]) - e0[i] <= 0 and -(x[i]-p[i]) - e0[i] <= 0
    #   deriv:     Dk(x)[i] - ek[i] <= 0    and -Dk(x)[i] - ek[i] <= 0
    n_con = 2 * (n + n1 + n2 + n3)
    A = lil_matrix((n_con, total))
    b = np.zeros(n_con)
    con = 0
    # Proximity rows (bias x toward p, breaks constant-path degeneracy).
    for i in range(n):
        A[con, off_x + i] = 1.0
        A[con, off_e0 + i] = -1.0
        b[con] = p[i]
        con += 1
        A[con, off_x + i] = -1.0
        A[con, off_e0 + i] = -1.0
        b[con] = -p[i]
        con += 1
    # Derivative rows.
    for order, e_off in ((1, off_e1), (2, off_e2), (3, off_e3)):
        for local_i, (i, terms) in enumerate(_derivative_rows(n, order)):
            for col, coeff in terms:
                A[con, off_x + col] = coeff
            A[con, e_off + local_i] = -1.0
            con += 1
            for col, coeff in terms:
                A[con, off_x + col] = -coeff
            A[con, e_off + local_i] = -1.0
            con += 1

    A_csr = csr_matrix(A)

    # Bounds: x in [max(lo, p-r), min(hi, p+r)]; slacks >= 0.
    bounds: List[Tuple[float, Optional[float]]] = []
    for i in range(n):
        x_lo = max(lo, p[i] - r[i])
        x_hi = min(hi, p[i] + r[i])
        if x_lo > x_hi:  # band fell outside [lo,hi]; collapse to the nearer edge
            x_lo = x_hi = float(min(hi, max(lo, p[i])))
        bounds.append((x_lo, x_hi))
    bounds.extend([(0.0, None)] * (n + n1 + n2 + n3))

    try:
        res = linprog(c, A_ub=A_csr, b_ub=b, bounds=bounds, method="highs")
    except Exception:  # pragma: no cover - solver blow-up guard
        return None
    if not res.success or res.x is None:
        return None
    return [float(v) for v in res.x[off_x:off_x + n]]


def parse_weights(spec: str, default: Tuple[float, float, float] = (1.0, 10.0, 100.0)) -> Tuple[float, float, float]:
    """Parse a ``"w1,w2,w3"`` config string into a weight tuple."""
    try:
        parts = [float(v) for v in str(spec).split(",")]
        if len(parts) == 3 and all(p >= 0 for p in parts):
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return default
