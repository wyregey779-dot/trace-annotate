"""
Cold-Storage Staging & Safety Manager (R1-R5 Compliant).
Enforces zero-touch read-only source protocols, chunked staging batches,
boundary escape prevention, directory overlap guards, and bidirectional head1mb_md5 + size verification.
"""
from __future__ import annotations
import os
import shutil
from typing import List, Generator, Tuple, Optional, Dict, Set
from trace_annotate.config import StagingConfig
from trace_annotate.core.hasher import head1mb_md5


class ColdStorageStagingError(RuntimeError):
    """Raised when cold storage safety violations or checksum mismatches occur."""
    pass


class StagingSecurityError(PermissionError, ValueError):
    """Raised when path traversal, symlink escape, or directory overlap is detected."""
    pass


def resolve_safe_source_path(source_root: str, rel_path: str) -> str:
    """Resolve relative path against source_root and guarantee it cannot escape source boundary."""
    source_root = os.path.realpath(os.path.abspath(source_root))
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
        raise StagingSecurityError(f"Path traversal detected: absolute paths not permitted: {rel_path}")
    full_path = os.path.realpath(os.path.abspath(os.path.join(source_root, rel_path)))
    src_prefix = source_root.rstrip(os.sep) + os.sep
    if not (full_path + os.sep).startswith(src_prefix) and full_path != source_root:
        raise StagingSecurityError(
            f"Path traversal detected: '{rel_path}' attempts to escape source directory '{source_root}'"
        )
    return full_path


def resolve_safe_staging_path(staging_root: str, rel_path: str) -> str:
    """Resolve relative path against staging_root and guarantee it cannot escape staging boundary."""
    staging_root = os.path.realpath(os.path.abspath(staging_root))
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
        raise StagingSecurityError(f"Path traversal detected: absolute paths not permitted: {rel_path}")
    full_path = os.path.abspath(os.path.join(staging_root, rel_path))
    stg_prefix = staging_root.rstrip(os.sep) + os.sep
    if not (full_path + os.sep).startswith(stg_prefix) and full_path != staging_root:
        raise StagingSecurityError(
            f"Path traversal detected: '{rel_path}' attempts to escape staging directory '{staging_root}'"
        )
    return full_path


class StagingManager:
    """
    Manages safe staged copies of media assets from external/read-only drives.
    Ensures:
      1. Source disk is NEVER written to.
      2. Path traversal, symlink escapes, and directory overlap are rejected.
      3. Files are copied in small batches with dynamic byte quota checks.
      4. Staged copies are verified against source (size + head1mb_md5).
      5. Staged copies are purged immediately unless explicitly marked in-use.
      6. Minimum free space on staging disk is strictly guarded.
    """

    def __init__(
        self,
        source_root: str,
        staging_root: str,
        config: Optional[StagingConfig] = None
    ):
        self.source_root = os.path.realpath(os.path.abspath(source_root))
        self.staging_root = os.path.realpath(os.path.abspath(staging_root))
        self.config = config or StagingConfig()
        self._in_use_locks: Set[str] = set()

        self._validate_directories()
        os.makedirs(self.staging_root, exist_ok=True)

    def _validate_directories(self) -> None:
        """Verify source and staging directories do not overlap."""
        if self.source_root == self.staging_root:
            raise StagingSecurityError(
                f"Staging root cannot be identical to source root: {self.source_root}"
            )
        # Check parent-child overlap
        src_with_sep = self.source_root.rstrip(os.sep) + os.sep
        stg_with_sep = self.staging_root.rstrip(os.sep) + os.sep
        if stg_with_sep.startswith(src_with_sep):
            raise StagingSecurityError(
                f"Staging root cannot be inside source root ({self.staging_root} inside {self.source_root})"
            )
        if src_with_sep.startswith(stg_with_sep):
            raise StagingSecurityError(
                f"Source root cannot be inside staging root ({self.source_root} inside {self.staging_root})"
            )

    def resolve_safe_source_path(self, rel_path: str) -> str:
        """Resolve relative path against source_root and guarantee it cannot escape source boundary."""
        return resolve_safe_source_path(self.source_root, rel_path)

    def resolve_safe_staging_path(self, rel_path: str) -> str:
        """Resolve relative path against staging_root and guarantee it cannot escape staging boundary."""
        return resolve_safe_staging_path(self.staging_root, rel_path)

    def lock_in_use(self, rel_path: str) -> None:
        """Mark a staged file as in-use to prevent premature cleanup."""
        self._in_use_locks.add(rel_path.replace("\\", "/"))

    def unlock_in_use(self, rel_path: str) -> None:
        """Release in-use lock on a staged file."""
        self._in_use_locks.discard(rel_path.replace("\\", "/"))

    def check_free_space(self, estimated_bytes: int = 0) -> None:
        """Verify that staging disk has enough free space according to guard threshold + batch size."""
        os.makedirs(self.staging_root, exist_ok=True)
        drive = os.path.splitdrive(self.staging_root)[0] or self.staging_root
        total, used, free = shutil.disk_usage(drive)
        required = self.config.free_disk_guard_bytes + max(0, estimated_bytes)
        if free < required:
            raise ColdStorageStagingError(
                f"Insufficient disk space in staging directory ({free / (1024**3):.2f} GB free). "
                f"Required guard + batch allocation: {required / (1024**3):.2f} GB."
            )

    def cleanup_completed(self, dry_run: bool = False) -> List[str]:
        """
        Scans staging_root and removes any files not currently locked in _in_use_locks.
        Returns list of cleaned relative paths.
        """
        cleaned = []
        for root, dirs, files in os.walk(self.staging_root):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.staging_root)
                rel_norm = rel_path.replace("\\", "/")
                if rel_norm not in self._in_use_locks:
                    cleaned.append(rel_norm)
                    if not dry_run:
                        try:
                            os.remove(full_path)
                        except Exception:
                            pass
        return cleaned

    def stage_batch(
        self,
        rel_paths: List[str]
    ) -> Generator[Dict[str, str], None, None]:
        """
        Stage a list of files in chunks.
        Yields a dictionary mapping rel_path -> staged_absolute_path.
        Automatically cleans up the staged batch after the caller finishes with each chunk,
        unless an item is explicitly marked in-use.
        """
        group_size = self.config.group_size

        for i in range(0, len(rel_paths), group_size):
            chunk = rel_paths[i : i + group_size]

            # Calculate estimated batch size
            batch_bytes = 0
            resolved_pairs: List[Tuple[str, str, str]] = []
            for rel in chunk:
                src_path = self.resolve_safe_source_path(rel)
                dst_path = self.resolve_safe_staging_path(rel)
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Source asset not found: {src_path}")
                batch_bytes += os.path.getsize(src_path)
                resolved_pairs.append((rel, src_path, dst_path))

            self.check_free_space(estimated_bytes=batch_bytes)

            staged_map: Dict[str, str] = {}
            created_dirs: Set[str] = set()

            try:
                for rel, src_path, dst_path in resolved_pairs:
                    parent_dir = os.path.dirname(dst_path)
                    os.makedirs(parent_dir, exist_ok=True)
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
                # Cleanup staged files unless locked in-use
                for rel, dst_path in staged_map.items():
                    rel_norm = rel.replace("\\", "/")
                    if rel_norm not in self._in_use_locks:
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
