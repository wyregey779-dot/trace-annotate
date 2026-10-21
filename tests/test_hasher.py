import os
import tempfile
import numpy as np
from PIL import Image
from trace_annotate.core.hasher import head1mb_md5, compute_dhash, hamming_distance


def test_head1mb_md5():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"a" * (2 * 1024 * 1024))
        name = f.name
    try:
        h1 = head1mb_md5(name)
        assert len(h1) == 32
    finally:
        os.remove(name)


def test_dhash_and_hamming():
    # Identical arrays -> identical dHash
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = 255
    h1 = compute_dhash(arr)
    h2 = compute_dhash(arr)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Slight modification -> small hamming distance
    arr_mod = arr.copy()
    arr_mod[10:15, 10:15] = 128
    h3 = compute_dhash(arr_mod)
    assert hamming_distance(h1, h3) <= 10
