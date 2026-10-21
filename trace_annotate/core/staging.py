"""
Cold-Storage Staging & Safety Manager (R1-R5 Compliant).
Enforces zero-touch read-only source protocols, chunked staging batches,
free disk space guards, and bidirectional head1mb_md5 + size verification.
"""
from __future__ import annotations
import os
import shutil
from typing import List, Generator, Tuple, Optional, Dict
from trace_annotate.config import StagingConfig
from trace_annotate.core.hasher import head1mb_md5


class ColdStorageStagingError(RuntimeError):
    """Raised when cold storage safety violations or checksum mismatches occur."""
    pass


class StagingManager:
    """
    Manages safe staged copies of media assets from external/read-only drives.
    Ensures:
      1. Source disk is NEVER written to.
      2. Files are copied in small, fixed-size batches (default 4).
      3. Staged copies are verified against source (size + head1mb_md5).
      4. Staged copies are purged immediately after batch processing.
      5. Minimum free space on staging disk is strictly guarded.
    """

    def __init__(
        self,
        source_root: str,
        staging_root: str,
        config: Optional[StagingConfig] = None
    ):
        self.source_root = os.path.abspath(source_root)
        self.staging_root = os.path.abspath(staging_root)
        self.config = config or StagingConfig()
        os.makedirs(self.staging_root, exist_ok=True)

    def check_free_space(self) -> None:
        """Verify that staging disk has enough free space according to guard threshold."""
        os.makedirs(self.staging_root, exist_ok=True)
        # Check disk usage using drive root or staging path
        drive = os.path.splitdrive(self.staging_root)[0] or self.staging_root
        total, used, free = shutil.disk_usage(drive)
        if free < self.config.free_disk_guard_bytes:
            raise ColdStorageStagingError(
                f"Insufficient disk space in staging directory ({free / (1024**3):.2f} GB free). "
                f"Required guard: {self.config.free_disk_guard_bytes / (1024**3):.2f} GB."
            )

    def stage_batch(
        self,
        rel_paths: List[str]
    ) -> Generator[Dict[str, str], None, None]:
        """
        Stage a list of files in chunks.
        Yields a dictionary mapping rel_path -> staged_absolute_path.
        Automatically cleans up the staged batch after the caller finishes with each chunk.
        """
        group_size = self.config.group_size

        for i in range(0, len(rel_paths), group_size):
            chunk = rel_paths[i : i + group_size]
            self.check_free_space()

            staged_map: Dict[str, str] = {}
            created_dirs: set[str] = set()

            try:
                for rel in chunk:
                    src_path = os.path.join(self.source_root, rel)
                    dst_path = os.path.join(self.staging_root, rel)

                    if not os.path.exists(src_path):
                        raise FileNotFoundError(f"Source asset not found: {src_path}")

                    parent_dir = os.path.dirname(dst_path)
                    os.makedirs(parent_dir, exist_ok=True)
                    # Only track subdirectories inside staging_root, never staging_root itself
                    if os.path.abspath(parent_dir) != self.staging_root and parent_dir.startswith(self.staging_root):
                        created_dirs.add(parent_dir)

                    # Copy to staging
                    shutil.copy2(src_path, dst_path)

                    # Double verify size and head1mb_md5
                    if self.config.verify_checksum:
                        src_size = os.path.getsize(src_path)
                        dst_size = os.path.getsize(dst_path)
                        if src_size != dst_size:
                            raise ColdStorageStagingError(
                                f"Staging size mismatch for {rel}: {src_size} != {dst_size}"
                            )

                        src_h1m = head1mb_md5(src_path)
                        dst_h1m = head1mb_md5(dst_path)
                        if src_h1m != dst_h1m:
                            raise ColdStorageStagingError(
                                f"Staging head1mb_md5 mismatch for {rel}: {src_h1m} != {dst_h1m}"
                            )

                    staged_map[rel] = dst_path

                yield staged_map

            finally:
                # Cleanup staged files immediately to keep local disk footprint minimal
                for rel, dst_path in staged_map.items():
                    if os.path.exists(dst_path):
                        try:
                            os.remove(dst_path)
                        except Exception:
                            pass

                # Clean empty child directories, never staging_root
                for d in sorted(created_dirs, key=len, reverse=True):
                    try:
                        if os.path.abspath(d) != self.staging_root and os.path.exists(d) and not os.listdir(d):
                            os.rmdir(d)
                    except Exception:
                        pass
