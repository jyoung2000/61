"""to_mono_float32 coercion: numpy shapes, NaN/Inf, and torch-tensor-like input."""

from __future__ import annotations

import numpy as np

from inflect.models.engine_base import to_mono_float32


def test_passthrough_1d():
    a = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    out = to_mono_float32(a)
    assert out.dtype == np.float32
    assert out.shape == (3,)


def test_channel_first_2d_to_mono():
    # (channels, n) with channels <= 8 and < n -> average over channels.
    a = np.ones((1, 100), dtype=np.float32)
    assert to_mono_float32(a).shape == (100,)
    stereo = np.stack([np.zeros(50), np.ones(50)]).astype(np.float32)  # (2, 50)
    out = to_mono_float32(stereo)
    assert out.shape == (50,)
    assert np.allclose(out, 0.5)


def test_channel_last_2d_to_mono():
    a = np.ones((100, 2), dtype=np.float32)  # (n, channels)
    assert to_mono_float32(a).shape == (100,)


def test_nan_inf_scrubbed():
    a = np.array([0.5, np.nan, np.inf, -np.inf], dtype=np.float32)
    out = to_mono_float32(a)
    assert np.all(np.isfinite(out))
    assert out[0] == np.float32(0.5)


class _FakeTensor:
    """Mimics the torch.Tensor methods to_mono_float32 relies on."""

    def __init__(self, arr):
        self._arr = arr
        self.moved = False

    def detach(self):
        return self

    def to(self, _device):
        self.moved = True
        return self

    def float(self):
        return self

    def numpy(self):
        return self._arr


def test_torch_tensor_like_path():
    fake = _FakeTensor(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))  # (1, 3)
    out = to_mono_float32(fake)
    assert fake.moved is True  # was moved to host
    assert out.shape == (3,)
    assert out.dtype == np.float32
