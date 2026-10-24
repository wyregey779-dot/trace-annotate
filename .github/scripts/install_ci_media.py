"""Install a digest-pinned Linux CI toolchain into a new runner-owned directory.

This is CI infrastructure, not an application installer. Extract only the two
named regular binary entries after verifying the complete archive SHA-256.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

URL = ('https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-23-13-04/'
       'ffmpeg-n9.0.2-35-g9a9b70bb8c-linux64-gpl-9.0.tar.xz')
SHA256 = '143e31ca53d65d95bde12cc861e331fd2275e48a6ca988e62c46e8f5b77b9811'
PREFIX = 'ffmpeg-n9.0.2-35-g9a9b70bb8c-linux64-gpl-9.0'


def main():
    output = Path(os.environ['RUNNER_TEMP']) / 'traceannotate-media-toolchain'
    output.mkdir(exist_ok=False)
    archive = output / 'toolchain.tar.xz'
    digest = hashlib.sha256()
    request = urllib.request.Request(URL, headers={'User-Agent': 'TraceAnnotate-CI'})
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, archive.open('xb') as target:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 200_000_000:
                raise RuntimeError('Toolchain archive exceeds its resource limit')
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError('Toolchain SHA-256 mismatch; no binaries extracted or executed')
    binaries = output / 'bin'
    binaries.mkdir()
    versions = {}
    with tarfile.open(archive, 'r:xz') as bundle:
        for name in ('ffmpeg', 'ffprobe'):
            member = bundle.getmember(PREFIX + '/bin/' + name)
            if not member.isfile() or member.size > 500_000_000:
                raise RuntimeError('Unexpected binary archive entry')
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError('Missing binary content')
            path = binaries / name
            with source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
            path.chmod(0o755)
            result = subprocess.run([str(path), '-version'], capture_output=True, text=True, check=True, timeout=30)
            versions[name] = result.stdout.splitlines()[0]
            if not versions[name].startswith(name + ' version n9.0.2-35-g9a9b70bb8c'):
                raise RuntimeError('Unexpected toolchain version: ' + versions[name])
    record = {'url': URL, 'sha256': SHA256, 'bytes': total, 'versions': versions,
              'provider': 'BtbN/FFmpeg-Builds; GPL binaries are CI-only, not redistributed with TraceAnnotate'}
    (output / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with Path(os.environ['GITHUB_PATH']).open('a', encoding='utf-8') as environment:
        environment.write(str(binaries) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
