"""
Hashing & Perceptual Fingerprinting Utilities.
Includes cold-storage fast validation (head1mb_md5), perceptual dHash,
and cross-platform image decoding safe against Chinese/Unicode paths and HEIC files.
"""
from __future__ import annotations
import os
import hashlib
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False


def head1mb_md5(file_path: str) -> str:
    """
    Compute MD5 of the first 1MB (1,048,576 bytes) of a file.
    Provides O(1) lightning-fast file identification without reading multi-gigabyte videos.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        chunk = f.read(1024 * 1024)
        hasher.update(chunk)
    return hasher.hexdigest()


def full_md5(file_path: str, chunk_size: int = 65536) -> str:
    """Compute full MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Safely load an image array (BGR) from disk.
    Fixes cv2.imread failing on Windows Unicode / Chinese paths.
    Falls back to Pillow / pillow-heif for HEIC files.
    """
    if not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()

    # For standard formats with OpenCV available
    if HAS_CV2 and ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        try:
            buf = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

    # Fallback to Pillow
    try:
        with Image.open(file_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img)
            # Convert RGB to BGR for OpenCV consistency
            return arr[:, :, ::-1].copy()
    except Exception:
        return None


def compute_dhash(img_or_path: np.ndarray | str, hash_size: int = 8) -> Optional[str]:
    """
    Compute difference hash (dHash) of an image.
    Outputs a hex string representation of the 64-bit perceptual fingerprint.
    """
    if isinstance(img_or_path, str):
        img = safe_load_image(img_or_path)
    else:
        img = img_or_path

    if img is None:
        return None

    # Convert to grayscale
    if len(img.shape) == 3:
        if HAS_CV2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        else:
            pil_img = Image.fromarray(img[:, :, ::-1]).convert("L")
            pil_resized = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            resized = np.array(pil_resized)
    else:
        if HAS_CV2:
            resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        else:
            pil_img = Image.fromarray(img).convert("L")
            pil_resized = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            resized = np.array(pil_resized)

    # Compare adjacent pixels: col[i] > col[i+1]
    diff = resized[:, 1:] > resized[:, :-1]
    # Flatten to integer
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)

    return f"{hash_val:016x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute hamming distance between two hex dHash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    x = int(hash1, 16) ^ int(hash2, 16)
    return bin(x).count("1")


def generate_thumbnail(
    img_or_path: np.ndarray | str,
    output_path: str,
    max_edge: int = 320,
    quality: int = 85
) -> bool:
    """
    Generate a normalized JPEG thumbnail with max edge limit.
    Saves via binary write to guarantee Chinese path safety.
    """
    if isinstance(img_or_path, str):
        img = safe_load_image(img_or_path)
    else:
        img = img_or_path

    if img is None:
        return False

    h, w = img.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    if HAS_CV2:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    else:
        rgb = img[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pil_img.save(output_path, format="JPEG", quality=quality)
        return True
