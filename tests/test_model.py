"""Smoke tests for the keyword-spotting model and data handling.

Run with: python -m pytest
"""

import numpy as np
import torch

from model import build_model
from train import to_spectrogram_nchw


def test_kws_forward_shape():
    model = build_model(in_channels=1, num_classes=3)
    # (batch, channel, frames, columns) e.g. an MFE spectrogram 99 x 40.
    x = torch.randn(8, 1, 99, 40)
    out = model(x)
    assert out.shape == (8, 3)


def test_to_spectrogram_from_flat():
    # 5 windows of 99 frames x 40 columns = 3960 MFE features.
    flat = np.zeros((5, 3960), dtype=np.float32)
    reshaped = to_spectrogram_nchw(flat, columns=40)
    assert reshaped.shape == (5, 1, 99, 40)


def test_to_spectrogram_row_major_order():
    # One window, 2 frames, 3 columns, row-major [f0c0, f0c1, f0c2, f1c0, ...].
    flat = np.array([[0, 1, 2, 10, 11, 12]], dtype=np.float32)
    reshaped = to_spectrogram_nchw(flat, columns=3)
    np.testing.assert_array_equal(reshaped[0, 0, 0], [0, 1, 2])
    np.testing.assert_array_equal(reshaped[0, 0, 1], [10, 11, 12])


def test_to_spectrogram_from_3d():
    x = np.zeros((4, 99, 40), dtype=np.float32)
    reshaped = to_spectrogram_nchw(x, columns=40)
    assert reshaped.shape == (4, 1, 99, 40)


def test_to_spectrogram_bad_divisor():
    flat = np.zeros((2, 101), dtype=np.float32)
    try:
        to_spectrogram_nchw(flat, columns=40)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-divisible feature count")
