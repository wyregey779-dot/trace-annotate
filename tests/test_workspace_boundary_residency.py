"""Dense boundary windows retain small exact features, never full decoded images."""
from __future__ import annotations

import gc
from pathlib import Path
import shutil
import subprocess
import weakref

import numpy as np
import pytest

from trace_annotate.extractors import shot_boundaries as SB
from trace_annotate.workspace import vision_jobs as V
from trace_annotate.workspace.store import WorkspaceStore


def original_features(frame):
    # Golden pre-optimization calculation, including coefficient order and
    # point-sampling positions. Do not replace this with the helper under test.
    gray = .114 * frame[..., 0] + .587 * frame[..., 1] + .299 * frame[..., 2]
    gray = gray[::max(1, frame.shape[0] // 64), ::max(1, frame.shape[1] // SB.ANALYSIS_WIDTH)]
    values, _ = np.histogram(gray, bins=32, range=(0, 255))
    total = values.sum()
    hist = (np.convolve(values.astype(np.float64) / total, [.25, .5, .25], mode='same')
            if total else values.astype(np.float64))
    return {'gray': gray, 'luma': float(gray.mean()), 'hist': hist}


def test_dense_sample_releases_full_hd_arrays_and_preserves_frame_identity(monkeypatch):
    references = []
    def decode(path):
        index = int(path.stem)
        # Every decoded source is separately allocated; weak references detect
        # accidental full-frame ownership, regardless of array nbytes claims.
        image = np.full((1080, 1920, 3), index * 27 + 13, dtype=np.uint8)
        references.append(weakref.ref(image))
        return image
    monkeypatch.setattr(V, '_load_frame', decode)
    samples = []
    for index in range(7):
        item = {'frame_index': str(index), 'source_pts': str(index * 1001),
                'proxy_pts': str(index * 40), 'path': str(Path('unused') / f'{index}.png')}
        sample = V._boundary_sample(item, index / 25)
        assert set(sample) == {'frame_index', 'seconds', 'source_pts', 'proxy_pts', 'gray', 'luma', 'hist'}
        assert sample['frame_index'] == item['frame_index'] and sample['source_pts'] == item['source_pts']
        expected = original_features(np.full((1080, 1920, 3), index * 27 + 13, dtype=np.uint8))
        np.testing.assert_array_equal(sample['gray'], expected['gray'])
        np.testing.assert_array_equal(sample['hist'], expected['hist'])
        assert sample['luma'] == expected['luma']
        samples.append(sample)
    gc.collect()
    assert all(ref() is None for ref in references), 'dense windows retained full decoded arrays'
    assert sum(item['gray'].nbytes + item['hist'].nbytes for item in samples) < 512 * 1024
    raw = [{**{key: sample[key] for key in ('frame_index', 'seconds', 'source_pts', 'proxy_pts')},
            'frame': np.full((1080, 1920, 3), index * 27 + 13, dtype=np.uint8)}
           for index, sample in enumerate(samples)]
    assert SB.detect_shot_boundaries(samples) == SB.detect_shot_boundaries(raw)
    assert SB.classify_scene_cut(samples, candidate_frame=3) == SB.classify_scene_cut(raw, candidate_frame=3)


def test_overlapping_real_boundary_windows_receive_small_nonaliased_frame_features(tmp_path, monkeypatch):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and ffprobe required for real overlapping boundary windows')
    source = tmp_path / 'three-shots.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin',
                    '-f', 'lavfi', '-i', 'color=blue:s=64x64:r=10:d=0.6',
                    '-f', 'lavfi', '-i', 'color=red:s=64x64:r=10:d=0.6',
                    '-f', 'lavfi', '-i', 'color=lime:s=64x64:r=10:d=0.6',
                    '-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
                    '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
                   check=True, capture_output=True, timeout=60)
    project = WorkspaceStore(tmp_path / 'boundary-workspace')
    project.create()
    jobs = None
    measured = {}
    windows = []
    original_sample, original_detect = V._boundary_sample, V.detect_shot_boundaries
    def capture_sample(item, seconds):
        sample = original_sample(item, seconds)
        expected = original_features(V._load_frame(Path(item['path'])))
        assert 'frame' not in sample
        np.testing.assert_array_equal(sample['gray'], expected['gray'])
        np.testing.assert_array_equal(sample['hist'], expected['hist'])
        measured[item['frame_index']] = {'sample': sample, 'item': item, 'expected': expected}
        return sample
    def capture_window(samples):
        windows.append(samples)
        for sample in samples:
            expected = measured[sample['frame_index']]['expected']
            assert 'frame' not in sample
            np.testing.assert_array_equal(sample['gray'], expected['gray'])
            np.testing.assert_array_equal(sample['hist'], expected['hist'])
            assert sample['luma'] == expected['luma']
        result = original_detect(samples)
        raw = [{**{key: sample[key] for key in ('frame_index', 'seconds', 'source_pts', 'proxy_pts')},
                'frame': V._load_frame(Path(measured[sample['frame_index']]['item']['path']))}
               for sample in samples]
        assert result == original_detect(raw)
        return result
    monkeypatch.setattr(V, '_boundary_sample', capture_sample)
    monkeypatch.setattr(V, 'detect_shot_boundaries', capture_window)
    try:
        admitted = project.import_source(source)
        current = project.get_task(admitted['task_id'])
        jobs = V.VisionJobs(project, person_model_path='', landmark_model_path='', landmark_python='',
                            face_model_path='missing', clip_model_path='missing', vlm_model_path='missing')
        def empty_extractors(thresholds):
            jobs._extractors = {key: None for key in ('face', 'ocr', 'clip', 'vlm', 'person')}
        monkeypatch.setattr(jobs, '_load_extractors', empty_extractors)
        batch = jobs.submit(current['id'], command_id='boundary-residency',
                            expected_revision=current['revision'], expected_sha256=current['content_sha256'])
        assert jobs.run_pending()[0]['state'] == 'succeeded'
        result = jobs.result(batch['jobs'][0]['id'])
        assert result['boundary_sampling']['status'] == 'success'
        assert len(windows) >= 2, 'fixture did not exercise multiple boundary windows'
        first = {s['frame_index']: s for s in windows[0]}
        second = {s['frame_index']: s for s in windows[1]}
        overlap = first.keys() & second.keys()
        assert overlap, 'fixture did not exercise overlapping windows'
        for index in overlap:
            assert first[index]['gray'] is second[index]['gray']
            assert first[index]['source_pts'] == second[index]['source_pts']
        registered = {item['sha256']: item for item in result['evidence']}
        assert result['boundary_frames']
        for frame in result['boundary_frames']:
            descriptor = registered[frame['evidence_sha256']]
            assert V.stream_sha256(project.root / descriptor['relative_path']) == frame['evidence_sha256']
        assert project.get_task(current['id']) == current
    finally:
        if jobs is not None:
            jobs.close()
        project.close()
