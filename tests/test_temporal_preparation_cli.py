"""Public preparation/open commands against FFmpeg and a real loopback server."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

import pytest

SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('media preparation CLI tests require FFmpeg and ffprobe')
    path = tmp_path / 'source.mp4'
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                             'testsrc2=size=160x90:rate=5:duration=1', '-an', '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', str(path)], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return path


def cli(*arguments):
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    return subprocess.run([sys.executable, '-B', '-m', 'trace_annotate.cli', *map(str, arguments)],
                          cwd=SOURCE, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=60)


def server_roundtrip(*arguments):
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen([sys.executable, '-B', '-m', 'trace_annotate.cli', *map(str, arguments)],
                               cwd=SOURCE, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8')
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        url = pool.submit(process.stdout.readline).result(timeout=30).strip()
        assert url.startswith('http://127.0.0.1:') and url.endswith('/editor.html'), url
        with urlopen(url.replace('editor.html', 'project.json'), timeout=10) as response:
            manifest = json.load(response)
        request = Request(url.replace('editor.html', 'playback.mp4'), headers={'Range': 'bytes=0-15'})
        with urlopen(request, timeout=10) as response:
            assert response.status == 206 and len(response.read()) == 16
        return manifest
    finally:
        process.terminate()
        process.wait(timeout=10)
        pool.shutdown(wait=True)


def test_public_prepare_then_readonly_project_open(sample, tmp_path):
    original = hashlib.sha256(sample.read_bytes()).hexdigest()
    output = tmp_path / 'prepared'
    result = cli('video-prepare', sample, '-o', output)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['source']['sha256'] == original
    manifest = server_roundtrip('video-ui', '--project', output, '--no-browser')
    assert manifest['video'] == report['video']
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == original


def test_one_command_prepares_and_opens(sample, tmp_path):
    output = tmp_path / 'one-command'
    manifest = server_roundtrip('video-open', sample, '-o', output, '--no-browser')
    assert manifest['timeline']['frame_count'] == 5
    assert (output / 'annotations.json').is_file()


def test_existing_directory_is_not_overwritten(sample, tmp_path):
    output = tmp_path / 'keep'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('existing work', encoding='utf-8')
    result = cli('video-prepare', sample, '-o', output)
    assert result.returncode == 1
    assert sentinel.read_text(encoding='utf-8') == 'existing work'
    assert list(output.iterdir()) == [sentinel]


def test_preparation_failure_has_no_published_project(tmp_path):
    path = tmp_path / 'fake.mp4'
    path.write_bytes(b'not a video')
    output = tmp_path / 'absent'
    result = cli('video-prepare', path, '-o', output)
    assert result.returncode == 1 and 'Error:' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('port', ['-1', '65536', 'bad'])
def test_invalid_server_port_is_rejected_before_work(port):
    result = cli('video-ui', '--project', 'does-not-exist', '--port', port, '--no-browser')
    assert result.returncode == 2 and 'port' in result.stderr


def test_standalone_no_browser_prints_editor_path():
    result = cli('video-ui', '--no-browser')
    assert result.returncode == 0 and Path(result.stdout.strip()).is_file()


def test_standalone_cannot_silently_ignore_server_port():
    result = cli('video-ui', '--port', '8765', '--no-browser')
    assert result.returncode == 1 and '--project' in result.stderr
