"""Classic entry points preserve editor resources and API boundaries."""
import re

import pytest

from tests.test_workspace_server import api, http_video, request, assert_error
from trace_annotate.workspace.classic import classic_resource


@pytest.mark.parametrize("path,controls", [
    ("/tools/general/", [b'videoInput', b'jsonInput', b'undo', b'redo', b'exportVTT', b'reviewInput']),
    ("/tools/embodied/", [b'fileInput', b'expectedSteps', b'inpStep', b'computeIdle', b'txtEvidence']),
])
def test_classic_page_keeps_original_controls_and_scoped_csp(api, path, controls):
    server, store = api
    before = store.get_snapshot()
    status, headers, html = request(server, "GET", path, authorized=False)
    assert status == 200
    assert all(control in html for control in controls)
    assert server.token.encode() not in html
    assert 'nonce-' + server.nonce in headers['Content-Security-Policy']
    assert "media-src 'self' blob:" in headers['Content-Security-Policy']
    assert "worker-src blob:" in headers['Content-Security-Policy']
    assert "script-src 'unsafe-inline'" not in headers['Content-Security-Policy']
    assert all(b'nonce="' + server.nonce.encode() + b'"' in tag
               for tag in re.findall(rb'<script\b[^>]*>', html))
    assert b'Set-Cookie' not in str(headers).encode()
    assert store.get_snapshot() == before


def test_classic_hash_script_and_no_directory_server(api):
    server, _ = api
    status, headers, data = request(server, "GET", "/tools/general/media-hash.js", authorized=False)
    assert status == 200 and b'TraceMediaHash' in data
    assert "media-src 'self' blob:" not in headers['Content-Security-Policy']
    for path in ("/tools/general/../server.py", "/tools/embodied/media-hash.js", "/tools/general/editor.html"):
        assert request(server, "GET", path, authorized=False)[0] == 404
    assert classic_resource('/tools/general/../../secret', server.nonce) is None


def test_classic_navigation_does_not_authorize_api_or_cross_origin(api):
    server, _ = api
    assert request(server, "GET", "/tools/general/", authorized=False)[0] == 200
    assert_error(request(server, "GET", "/api/v1/projects/current/snapshot", authorized=False), 403, "INVALID_TOKEN")
    assert_error(request(server, "GET", "/tools/general/", authorized=False,
                         headers={"Origin": "https://example.org"}), 403, "FORBIDDEN_ORIGIN")
    assert_error(request(server, "GET", "/tools/general/media-hash.js", authorized=False,
                         headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "script"}),
                 403, "FORBIDDEN_ORIGIN")
    status, _, _ = request(server, "GET", "/tools/general/", authorized=False,
                           headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate"})
    assert status == 200


def test_classic_bootstrap_csp_does_not_relax_workspace(api):
    server, _ = api
    request(server, "GET", "/tools/general/", authorized=False)
    status, headers, _ = request(server, "GET", "/", authorized=False)
    assert status == 200
    assert "media-src 'self' blob:" not in headers['Content-Security-Policy']
    assert "worker-src blob:" not in headers['Content-Security-Policy']
