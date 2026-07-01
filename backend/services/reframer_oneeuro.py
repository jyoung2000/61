"""One-Euro filter for the reframer target-x trajectory.

The planner historically smoothed its per-sample target position with a
fixed-band velocity-adaptive EMA (``reframer_planner._ema_target``): pick a
smoothing alpha per distance band and live with the lag/jitter tradeoff that
alpha bakes in.

The One-Euro filter (Casiez, Roussel & Vogel, CHI 2012 — "1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems") removes
that fixed tradeoff by adapting its cutoff frequency to the signal speed:

  * when the subject is nearly still, the cutoff is low → heavy smoothing,
    jitter is crushed;
  * when the subject moves fast, the cutoff rises → little smoothing, the
    crop keeps up with almost no lag.

This is exactly the "smooth AND snappy" behaviour a human camera operator
produces, in ~30 lines with no dependencies beyond ``math``.

The implementation is deliberately framework-free and side-effect-free so it
can be unit-tested in isolation (see ``tests/test_reframer_oneeuro.py``) and
reused by any 1-D signal (target-x, and later the zoom/scale term).
"""

from __future__ import annotations

import math
from typing import Optional


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low-pass at ``cutoff`` Hz.

    ``dt`` is the sample period in seconds. Derived from the standard
    exponential-smoothing / RC low-pass relation ``alpha = 1/(1 + tau/dt)``
    with ``tau = 1/(2*pi*cutoff)``.
    """
    if dt <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    """First-order low-pass with an externally supplied alpha per step."""

    __slots__ = ("_y", "_has_prev")

    def __init__(self) -> None:
        self._y: float = 0.0
        self._has_prev: bool = False

    @property
    def last(self) -> float:
        return self._y

    @property
    def initialized(self) -> bool:
        return self._has_prev

    def filt(self, x: float, alpha: float) -> float:
        if not self._has_prev:
            self._y = x
            self._has_prev = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    def reset(self) -> None:
        self._has_prev = False
        self._y = 0.0


class OneEuroFilter:
    """Speed-adaptive low-pass filter for a scalar signal.

    Parameters
    ----------
    freq:
        Nominal sample rate (Hz). Only used before the first real ``dt`` is
        known; subsequent calls pass an explicit timestamp.
    mincutoff:
        Cutoff (Hz) at zero speed. Lower → smoother/laggier when the subject
        holds still. Typical 0.5–1.5.
    beta:
        Speed coefficient. Higher → the cutoff opens up faster as the subject
        accelerates → snappier tracking. Typical 0.005–0.05 for pixel signals.
    dcutoff:
        Cutoff (Hz) of the low-pass applied to the derivative estimate.
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.02,
        dcutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq must be > 0")
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        """Forget all history — the next sample seeds the filter fresh.

        Used on a hard cut / speaker switch, where we want the crop to snap
        to the new position instead of easing from the old one.
        """
        self._x.reset()
        self._dx.reset()
        self._last_t = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """Filter one sample.

        ``t`` is an absolute timestamp in seconds. When omitted, the nominal
        ``freq`` sets the sample period. Non-monotonic or zero ``dt`` falls
        back to ``1/freq`` so the filter never divides by zero.
        """
        x = float(x)
        if t is None:
            dt = 1.0 / self.freq
        elif self._last_t is None:
            dt = 1.0 / self.freq
        else:
            dt = t - self._last_t
            if dt <= 0:
                dt = 1.0 / self.freq
        self._last_t = t

        # Derivative of the signal, low-passed at a fixed cutoff.
        if self._x.initialized:
            dx = (x - self._x.last) / dt
        else:
            dx = 0.0
        edx = self._dx.filt(dx, _alpha(self.dcutoff, dt))

        # Speed-adaptive cutoff, then low-pass the raw signal with it.
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x.filt(x, _alpha(cutoff, dt))

    @property
    def initialized(self) -> bool:
        return self._x.initialized

    @property
    def last(self) -> float:
        return self._x.last
