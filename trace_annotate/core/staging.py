"""Batch media copies into a dedicated staging directory and clean completed files.

Paths are checked against source/staging boundaries. Copy verification compares
size and the first MiB hash. Use staging storage controlled by the caller: these
path checks do not provide atomic protection against concurrent directory swaps.
"""
from __future__ import annotations
import os
import shutil
import stat
import tempfile
from typing import List, Generator, Tuple, Optional, Dict, Set
from trace_annotate.config import StagingConfig
from trace_annotate.core.hasher import head1mb_md5


class ColdStorageStagingError(RuntimeError):
    """Raised when cold storage safety violations or checksum mismatches occur."""
    pass


class StagingSecurityError(PermissionError, ValueError):
    """Raised when path traversal, symlink escape, or directory overlap is detected."""
    pass


def _is_within(root: str, path: str) -> bool:
    """Compare canonical paths with the platform's case and drive semantics."""
    root = os.path.normcase(os.path.realpath(root))
    path = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def resolve_safe_source_path(source_root: str, rel_path: str) -> str:
    """Resolve a relative source path and reject targets outside source_root."""
    source_root = os.path.realpath(os.path.abspath(source_root))
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
        raise StagingSecurityError(f"Path traversal detected: absolute paths not permitted: {rel_path}")
    full_path = os.path.realpath(os.path.abspath(os.path.join(source_root, rel_path)))
    if not _is_within(source_root, full_path):
        raise StagingSecurityError(
            f"Path traversal detected: '{rel_path}' attempts to escape source directory '{source_root}'"
        )
    return full_path


def resolve_safe_staging_path(staging_root: str, rel_path: str) -> str:
    """Resolve a relative staging path and reject targets outside staging_root."""
    staging_root = os.path.realpath(os.path.abspath(staging_root))
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
        raise StagingSecurityError(f"Path traversal detected: absolute paths not permitted: {rel_path}")
    # realpath follows both POSIX links and Windows directory junctions. A
    # lexical abspath check alone allows copy and cleanup to affect other disks.
    full_path = os.path.realpath(os.path.abspath(os.path.join(staging_root, rel_path)))
    if not _is_within(staging_root, full_path):
        raise StagingSecurityError(
            f"Path traversal detected: '{rel_path}' attempts to escape staging directory '{staging_root}'"
        )
    return full_path


class StagingManager:
    """
    Read source assets and manage copies in a separate, caller-controlled directory.

    Rejects overlapping roots and paths resolving outside their root, checks free
    space before each batch, and optionally verifies size and first-MiB hashes.
    Cleanup honors in-use locks, skips directory links and reports removal errors.
    A free-space check is not a disk reservation; callers must coordinate access
    to staging and keep source backups independently.
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
        # Keep both the requested entry and its target. This protects a regular
        # file locked through an in-tree alias without deleting through that alias.
        self._in_use_locks: Dict[str, str] = {}

        self._validate_directories()
        os.makedirs(self.staging_root, exist_ok=True)

    def _validate_directories(self) -> None:
        """Verify source and staging directories do not overlap."""
        if os.path.normcase(self.source_root) == os.path.normcase(self.staging_root):
            raise StagingSecurityError(
                f"Staging root cannot be identical to source root: {self.source_root}"
            )
        # Check parent-child overlap
        if _is_within(self.source_root, self.staging_root):
            raise StagingSecurityError(
                f"Staging root cannot be inside source root ({self.staging_root} inside {self.source_root})"
            )
        if _is_within(self.staging_root, self.source_root):
            raise StagingSecurityError(
                f"Source root cannot be inside staging root ({self.source_root} inside {self.staging_root})"
            )

    def resolve_safe_source_path(self, rel_path: str) -> str:
        """Resolve a relative source path and reject targets outside source_root."""
        return resolve_safe_source_path(self.source_root, rel_path)

    def resolve_safe_staging_path(self, rel_path: str) -> str:
        """Resolve a relative staging path and reject targets outside staging_root."""
        return resolve_safe_staging_path(self.staging_root, rel_path)

    def lock_in_use(self, rel_path: str) -> None:
        """Mark a staged file as in-use to prevent premature cleanup."""
        target = self.resolve_safe_staging_path(rel_path)
        self._in_use_locks[self._lock_key(rel_path)] = os.path.normcase(target)

    def unlock_in_use(self, rel_path: str) -> None:
        """Release in-use lock on a staged file."""
        self._in_use_locks.pop(self._lock_key(rel_path), None)

    def _entry_path(self, rel_path: str) -> str:
        """Validate a relative directory entry without dereferencing its leaf."""
        if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
            raise StagingSecurityError(f"Absolute staging paths are not permitted: {rel_path}")
        entry = os.path.abspath(os.path.join(self.staging_root, rel_path))
        root_key, entry_key = os.path.normcase(self.staging_root), os.path.normcase(entry)
        try:
            contained = os.path.commonpath([root_key, entry_key]) == root_key
        except ValueError:
            contained = False
        if not contained or entry_key == root_key:
            raise StagingSecurityError(f"Invalid staging file path: {rel_path}")
        return entry

    def _lock_key(self, rel_path: str) -> str:
        return os.path.normcase(self._entry_path(rel_path))

    def _is_in_use(self, rel_path: str, *, follow_leaf: bool = True) -> bool:
        entry = self._entry_path(rel_path)
        if os.path.normcase(entry) in self._in_use_locks:
            return True
        if not follow_leaf and os.path.islink(entry):
            return False
        target = os.path.normcase(self.resolve_safe_staging_path(rel_path))
        return target in self._in_use_locks.values()

    @staticmethod
    def _is_directory_link(path: str) -> bool:
        info = os.lstat(path)
        # Python 3.9 lacks os.path.isjunction(); reparse attributes cover Windows
        # junctions as well as directory symlinks without traversing them.
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )

    def _cleanup_entry_path(self, rel_path: str) -> str:
        entry = self._entry_path(rel_path)
        parent = os.path.dirname(entry)
        if os.path.normcase(os.path.realpath(parent)) != os.path.normcase(parent):
            raise StagingSecurityError(f"Staging cleanup parent is redirected: {rel_path}")
        # Retain the target boundary check, including for broken links. The
        # returned canonical target is deliberately NOT used for deletion.
        self.resolve_safe_staging_path(rel_path)
        return entry

    def _copy_destination_path(self, rel_path: str) -> str:
        """Require a direct staging entry so copies cannot replace a link target."""
        entry = self._cleanup_entry_path(rel_path)
        if os.path.islink(entry) or os.path.normcase(os.path.realpath(entry)) != os.path.normcase(entry):
            raise StagingSecurityError(f"Staging copy destination is redirected: {rel_path}")
        return entry

    @staticmethod
    def _remove_file(path: str) -> bool:
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ColdStorageStagingError(f"Could not remove staged file: {path}") from exc
        return True

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
        Remove unlocked files without traversing directory symlinks or junctions.

        In-tree file symlinks are unlinked as entries; their targets are not
        deleted through the link. Links escaping staging are rejected. Return
        successfully removed paths, or proposed paths when dry_run is true.
        """
        cleaned = []
        if os.path.normcase(os.path.realpath(self.staging_root)) != os.path.normcase(self.staging_root):
            raise StagingSecurityError("Staging root was redirected before cleanup")
        for root, dirs, files in os.walk(self.staging_root, followlinks=False):
            dirs[:] = [d for d in dirs if not self._is_directory_link(os.path.join(root, d))]
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.staging_root)
                rel_norm = rel_path.replace("\\", "/")
                full_path = self._cleanup_entry_path(rel_path)
                if not self._is_in_use(rel_path, follow_leaf=False):
                    if dry_run:
                        if os.path.lexists(full_path):
                            cleaned.append(rel_norm)
                    elif self._remove_file(full_path):
                        cleaned.append(rel_norm)
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
                dst_path = self._copy_destination_path(rel)
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Source asset not found: {src_path}")
                batch_bytes += os.path.getsize(src_path)
                resolved_pairs.append((rel, src_path, dst_path))

            self.check_free_space(estimated_bytes=batch_bytes)

            staged_map: Dict[str, str] = {}
            created_dirs: Set[str] = set()

            try:
                for rel, src_path, dst_path in resolved_pairs:
                    # Recheck after yielding a previous batch: staging may have
                    # been changed by another process between batches.
                    dst_path = self._copy_destination_path(rel)
                    parent_dir = os.path.dirname(dst_path)
                    os.makedirs(parent_dir, exist_ok=True)
                    if os.path.abspath(parent_dir) != self.staging_root and parent_dir.startswith(self.staging_root):
                        created_dirs.add(parent_dir)

                    # Copy to staging
                    dst_path = self._copy_destination_path(rel)
                    # Replace a completed private copy rather than writing
                    # through an existing hard link to an unrelated file.
                    fd, copy_path = tempfile.mkstemp(prefix=".trace-copy-", dir=parent_dir)
                    os.close(fd)
                    try:
                        shutil.copy2(src_path, copy_path)
                        checked_dst = self._copy_destination_path(rel)
                        if os.path.normcase(checked_dst) != os.path.normcase(dst_path):
                            raise StagingSecurityError(f"Staging path changed during copy: {rel}")
                        os.replace(copy_path, dst_path)
                    finally:
                        if os.path.exists(copy_path):
                            os.remove(copy_path)
                    staged_map[rel] = dst_path

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

                yield staged_map

            finally:
                # Cleanup staged files unless locked in-use
                for rel, dst_path in staged_map.items():
                    if not self._is_in_use(rel):
                        # Never follow a junction inserted while the caller
                        # was processing the yielded batch.
                        safe_path = self._cleanup_entry_path(rel)
                        resolved = self.resolve_safe_staging_path(rel)
                        if os.path.normcase(resolved) != os.path.normcase(dst_path):
                            raise StagingSecurityError(f"Staging path changed during processing: {rel}")
                        self._remove_file(safe_path)

                # Clean empty child directories, never staging_root
                for d in sorted(created_dirs, key=len, reverse=True):
                    try:
                        if os.path.abspath(d) != self.staging_root and os.path.exists(d) and not os.listdir(d):
                            os.rmdir(d)
                    except Exception:
                        pass
