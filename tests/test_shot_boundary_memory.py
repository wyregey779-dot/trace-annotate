"""Boundary luma extraction is bounded by the patch, not the source pixel count."""
import tracemalloc

import numpy as np
import pytest

from trace_annotate.extractors import shot_boundaries as SB


@pytest.mark.parametrize('channels', [1, 3])
def test_full_hd_luma_patch_does_not_allocate_full_resolution_float_planes(channels):
    shape = (1080, 1920) if channels == 1 else (1080, 1920, 3)
    frame = np.full(shape, 137, dtype=np.uint8)
    tracemalloc.start()
    try:
        patch = SB.frame_features(frame)['gray']
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, 'small patch extraction allocated full-size float planes'
    assert patch.shape == (68, 96)


@pytest.mark.parametrize('shape', [(513, 1025, 3), (127, 193), (32, 48, 3)])
def test_early_sampling_preserves_the_existing_luma_and_histogram(shape):
    frame = np.random.default_rng(31).integers(0, 256, shape, dtype=np.uint8)
    old_gray = (0.114 * frame[..., 0] + 0.587 * frame[..., 1] + 0.299 * frame[..., 2]
                if len(shape) == 3 else frame.astype(np.float64))
    old_patch = old_gray[::max(1, shape[0] // 64), ::max(1, shape[1] // SB.ANALYSIS_WIDTH)].astype(np.float64)
    actual = SB.frame_features(frame)
    np.testing.assert_array_equal(actual['gray'], old_patch)
    assert actual['luma'] == float(old_patch.mean())
    np.testing.assert_array_equal(actual['hist'], SB._histogram(old_patch))
