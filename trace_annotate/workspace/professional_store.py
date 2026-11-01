"""Transactional orchestration for professional task exchange and analysis.

The ordinary store command path owns revisions, conflicts and receipts. This
module prepares typed changes inside that transaction; it does not write around
the command journal or interpret imported filesystem paths.
"""
from __future__ import annotations

import copy
import json

from . import schema as S
from .errors import SchemaInvalid, WorkspaceError

# Leave room for the bounded command identity/revision envelope when a file
# produced by this exporter is returned through the1MiB HTTP command route.
EXPORT_DOCUMENT_LIMIT = 1024 * 1024 - 4096


def prepare_task(store, current, operation, payload, *, command_id):
    from . import professional_schema as PS
    from .professional_export import import_payload

    task = copy.deepcopy(current)
    media = store.get_media(task['media_id'])
    if operation == 'import_professional':
        if set(payload) != {'document'}:
            raise SchemaInvalid('SCHEMA_INVALID', 'Professional import requires one document')
        imported = import_payload(payload['document'], media, task['id'])
        task['annotations'] = imported['annotations']
        task['extensions'] = imported['extensions']
        task['legacy_document_metadata'] = imported['legacy_document_metadata']
        archive = task['extensions']['professional']['import_source']
        source_document = archive.pop('document')
        # Keep the original exchange document exactly as JSON in immutable
        # project storage. Embedding it in every later task/export recursively
        # doubles the file on repeated round trips.
        archive['document_artifact'] = store._put_artifact(S.canonical_json(source_document))
    elif operation == 'set_analysis_reports':
        reports = payload.get('reports')
        if (set(payload) != {'reports'} or not isinstance(reports, list)
                or not 1 <= len(reports) <= PS.MAX_REPORTS):
            raise SchemaInvalid('SCHEMA_INVALID', 'Analysis bundle requires a bounded nonempty report list')
        seen = set()
        for report in reports:
            PS.validate_analysis(report, media)
            if report['id'] in seen:
                raise SchemaInvalid('SCHEMA_INVALID', 'Analysis bundle contains duplicate report identities')
            seen.add(report['id'])
            task = prepare_task(store, task, 'set_analysis_report', {'report': report}, command_id=command_id)
    elif operation == 'set_analysis_report':
        if set(payload) != {'report'}:
            raise SchemaInvalid('SCHEMA_INVALID', 'Analysis command requires one report')
        report = copy.deepcopy(payload['report'])
        PS.validate_analysis(report, media)
        if len(S.canonical_json(report)) > 256 * 1024:
            raise SchemaInvalid('ANALYSIS_LIMIT_EXCEEDED', 'One report exceeds256KiB')
        if report.get('subject_annotation_id') is not None:
            subject = next((a for a in task['annotations'] if a['id'] == report['subject_annotation_id']), None)
            if subject is None or subject['type'] != 'segment':
                raise SchemaInvalid('ANALYSIS_SUBJECT_MISMATCH', 'Analysis subject is not a current segment')
            expected = {'start': subject['start']['seconds'], 'end': subject['end']['seconds']}
            if report['window'] != expected:
                raise SchemaInvalid('ANALYSIS_SUBJECT_MISMATCH', 'Analysis window differs from current boundaries')
        professional = task.setdefault('extensions', {}).setdefault('professional', {})
        if not isinstance(professional, dict):
            raise WorkspaceError('RECOVERY_REQUIRED', 'Stored professional analysis is invalid')
        reports = professional.setdefault('analysis_reports', [])
        if not isinstance(reports, list):
            raise WorkspaceError('RECOVERY_REQUIRED', 'Stored professional analysis is invalid')
        existing = next((r for r in reports if r['id'] == report['id']), None)
        if existing is not None and S.canonical_json(existing) != S.canonical_json(report):
            raise SchemaInvalid('ANALYSIS_ID_EXISTS', 'Report identities are immutable; publish a new report ID')
        if existing is None:
            if len(reports) >= PS.MAX_REPORTS:
                raise SchemaInvalid('ANALYSIS_LIMIT_EXCEEDED', 'Task analysis count exceeds256')
            reports.append(report)
    else:
        raise SchemaInvalid('UNSUPPORTED_OPERATION', 'Unknown professional command')
    # A task stays small enough for the same bounded HTTP command path and
    # recovery envelope. Large media belongs in immutable registered objects.
    if len(S.canonical_json(task)) > 768 * 1024:
        raise SchemaInvalid('PROFESSIONAL_LIMIT_EXCEEDED', 'Professional task exceeds768KiB')
    return task


def apply_import(store, task_id, *, command_id, expected_revision, expected_sha256, document, request_body):
    return store.apply_task_command(
        command_id=command_id, task_id=task_id, expected_revision=expected_revision,
        expected_sha256=expected_sha256, operation='import_professional',
        payload={'document': document}, request_body=request_body)['response']


def apply_analysis(store, task_id, *, command_id, expected_revision, expected_sha256, report, request_body):
    return store.apply_task_command(
        command_id=command_id, task_id=task_id, expected_revision=expected_revision,
        expected_sha256=expected_sha256, operation='set_analysis_report',
        payload={'report': report}, request_body=request_body)['response']


def export_professional(store, task_id):
    from .professional_export import build_document

    with store.transaction():
        task = store.get_task(task_id)
        media = store.get_media(task['media_id'])
        template = next(t for t in store.list_templates() if
                        (t['id'], t['version']) == (task['template_id'], task['template_version']))
        history = {'scope': 'available_local_audit_index',
                   'revisions': [],
                   'commits': [], 'receipts': [], 'source_envelopes': []}
        source_artifacts = {}
        for ident, revision, text, digest, commit_id, created_at in store.conn.execute(
                'SELECT id,revision,body,content_sha256,commit_id,created_at FROM task_revisions '
                'WHERE task_id=? ORDER BY revision', (task_id,)):
            body = json.loads(text)
            history['revisions'].append({'id': ident, 'task_id': task_id, 'revision': revision,
                'content_sha256': digest, 'annotation_count': len(body['annotations']),
                'commit_id': commit_id, 'created_at': created_at})
            source = body.get('extensions', {}).get('professional', {}).get('import_source', {})
            if 'document_artifact' in source:
                descriptor = source['document_artifact']
                source_artifacts[descriptor['id']] = descriptor
        # Audit metadata is portable. Complete old revision bodies and source
        # bytes remain in the project/backup, and are explicitly not embedded.
        history['source_envelopes'] = [
            {'document_artifact': descriptor, 'scope': 'external_project_artifact_not_embedded'}
            for _, descriptor in sorted(source_artifacts.items())]
        for body, in store.conn.execute('SELECT body FROM durable_commits ORDER BY rowid'):
            value = json.loads(body)
            if value.get('task_id') == task_id:
                history['commits'].append(value)
        for row in store.conn.execute('SELECT command_id,task_id,request_sha256,state,response,created_at '
                                      'FROM command_receipts WHERE task_id=? ORDER BY rowid', (task_id,)):
            value = dict(zip(('command_id', 'task_id', 'request_sha256', 'state', 'response', 'created_at'), row))
            value['response'] = json.loads(value['response']) if value['response'] else None
            history['receipts'].append(value)
        if store.task_digest(task) != task['content_sha256']:
            raise WorkspaceError('RECOVERY_REQUIRED', 'Task content identity is invalid')
        bindings_row = store.conn.execute('SELECT value FROM meta WHERE key=?',
                                          ('media_objects:' + media['id'],)).fetchone()
        bindings = json.loads(bindings_row[0]) if bindings_row else {}
        artifacts = []
        for ident in sorted({ref['id'] for ref in bindings.values()}):
            row = store.conn.execute('SELECT id,rel_path,sha256,size_bytes FROM artifacts WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise WorkspaceError('RECOVERY_REQUIRED', 'Professional export artifact is missing')
            artifacts.append({'id': row[0], 'relative_path': row[1], 'sha256': row[2], 'size_bytes': str(row[3])})
        known = {a['id'] for a in artifacts}
        for ident, descriptor in sorted(source_artifacts.items()):
            row = store.conn.execute('SELECT id,rel_path,sha256,size_bytes FROM artifacts WHERE id=?', (ident,)).fetchone()
            actual = ({'id': row[0], 'relative_path': row[1], 'sha256': row[2], 'size_bytes': str(row[3])}
                      if row is not None else None)
            if descriptor != actual:
                raise WorkspaceError('RECOVERY_REQUIRED', 'Professional source archive is missing or inconsistent')
            if ident not in known:
                artifacts.append(actual)
                known.add(ident)
        document = build_document(task, media, template, history=history, artifacts=artifacts)
        data = S.canonical_json(document)
        if len(data) > EXPORT_DOCUMENT_LIMIT:
            raise SchemaInvalid('PROFESSIONAL_LIMIT_EXCEEDED',
                                'Professional export exceeds its round-trip budget; keep the full project/backup')
        return {'data': data, 'content_type': 'application/json; charset=utf-8',
                'filename': f'TraceAnnotate-{task_id}-r{task["revision"]}.professional.json',
                'binding': {'task_id': task_id, 'revision': task['revision'],
                            'content_sha256': task['content_sha256']},
                'losses': 'Complete current task; audit index only, not historical revision bodies or archived source bytes. No media bytes. Preserve the full project/backup for history. Existing admitted source required for import.'}
