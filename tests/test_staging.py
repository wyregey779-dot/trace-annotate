import os
import shutil
import tempfile
import pytest
from trace_annotate.config import StagingConfig
from trace_annotate.core.staging import StagingManager, ColdStorageStagingError
from trace_annotate.core.hasher import head1mb_md5


def test_staging_manager_lifecycle():
    with tempfile.TemporaryDirectory() as temp_dir:
        src = os.path.join(temp_dir, "src")
        stg = os.path.join(temp_dir, "stg")
        os.makedirs(src, exist_ok=True)
        os.makedirs(stg, exist_ok=True)

        # Create dummy source files
        files = ["sub/a.txt", "sub/b.txt", "c.txt"]
        for f in files:
            p = os.path.join(src, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"Content for {f}")

        cfg = StagingConfig(group_size=2, free_disk_guard_bytes=1024)
        mgr = StagingManager(source_root=src, staging_root=stg, config=cfg)

        processed = []
        for batch in mgr.stage_batch(files):
            for rel, staged_path in batch.items():
                assert os.path.exists(staged_path)
                assert os.path.getsize(staged_path) == os.path.getsize(os.path.join(src, rel))
                assert head1mb_md5(staged_path) == head1mb_md5(os.path.join(src, rel))
                processed.append(rel)

        assert processed == files
        # Check that staged files were cleaned up
        for f in files:
            assert not os.path.exists(os.path.join(stg, f))
