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

URL = ('https://github.com/mrkbac/portable-ffmpeg/releases/download/ffmpeg-9.0.2-1/'
       'ffmpeg-9.0.2-linux-amd64.tar.xz')
SHA256 = 'f78d9e9833dc773cd45ef03f20368b8cd67fd353fd8e85bb61c6aabb69ef0148'
VERSION = 'n9.0.2-39-g7adfbe40dc-20261031'


def main():
    output = Path(os.environ['RUNNER_TEMP']) / 'traceannotate-media-toolchain'
    output.mkdir(exist_ok=False)
    archive = output / 'toolchain.tar.xz'
    digest = hashlib.sha256()
    request = urllib.request.Request(URL, headers={'User-Agent': 'TraceAnnotate-CI'})
    print('Downloading ' + URL, flush=True)
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
            member = bundle.getmember(name)
            if not member.isfile() or not 0 < member.size <= 500_000_000:
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
            if not versions[name].startswith(name + ' version ' + VERSION + ' '):
                raise RuntimeError('Unexpected toolchain version: ' + versions[name])
    record = {'url': URL, 'sha256': SHA256, 'bytes': total, 'versions': versions,
              'provider': 'portable-ffmpeg immutable release of BtbN/FFmpeg-Builds; CI-only'}
    (output / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with Path(os.environ['GITHUB_PATH']).open('a', encoding='utf-8') as environment:
        environment.write(str(binaries) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
