"""Bounded single-frame previews using the same frozen video render plan."""
from __future__ import annotations

import io
import re
from PIL import Image

from . import render_layout as L
from .errors import Conflict, LimitExceeded, SchemaInvalid, WorkspaceError

MAX_PREVIEW_BYTES = 32 * 1024 * 1024


class _BoundedPNG(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_PREVIEW_BYTES:
            raise LimitExceeded('LIMIT_EXCEEDED', 'Single-frame preview exceeds32MiB')
        return super().write(data)


def preview(server, task_id, body):
    fields = {'expected_revision', 'expected_sha256', 'recipe', 'frame_index'}
    if not isinstance(body, dict) or set(body) != fields:
        raise SchemaInvalid('SCHEMA_INVALID', 'Preview fields are missing or unknown')
    value = body['frame_index']
    if not isinstance(value, str) or re.fullmatch(r'0|[1-9][0-9]{0,18}', value) is None:
        raise SchemaInvalid('SCHEMA_INVALID', 'Preview frame index must be a canonical decimal string')
    if server.closing.is_set():
        raise Conflict('WORKSPACE_CLOSING', 'Workspace is closing')
    if not server.render_preview_lock.acquire(blocking=False):
        raise Conflict('PREVIEW_BUSY', 'Another preview is finishing; retry after it completes')
    try:
        plan = server.render_jobs.plan(task_id, expected_revision=body['expected_revision'],
            expected_sha256=body['expected_sha256'], recipe=body['recipe'])
        index = int(value)
        if index >= int(plan['frame_count']):
            raise SchemaInvalid('FRAME_OUT_OF_RANGE', 'Preview frame does not exist')
        image = server.image_service.image(plan['binding']['media_id'], index)
        if (image['row']['frame_index'] != value
                or image['playback_sha256'] != plan['binding']['playback_sha256']):
            raise WorkspaceError('RECOVERY_REQUIRED', 'Preview image differs from the frozen playback')
        with Image.open(io.BytesIO(image['data'])) as frame:
            if frame.size != tuple(plan['geometry']['source_rect'][2:]):
                raise WorkspaceError('RENDER_MEDIA_CHANGED', 'Preview geometry differs from the frozen plan')
            frame.load()
            output = L.composite_frame(plan, frame, index, font=server.render_jobs.trusted_font())
            try:
                with _BoundedPNG() as encoded:
                    output.save(encoded, format='PNG')
                    data = encoded.getvalue()
            finally:
                output.close()
        return {'data': data, 'content_type': 'image/png', 'frame_index': value,
            'plan_sha256': plan['content_sha256'],
            'binding': {'task_id': task_id, 'revision': plan['binding']['revision'],
                        'content_sha256': plan['binding']['task_content_sha256']}}
    finally:
        server.render_preview_lock.release()
