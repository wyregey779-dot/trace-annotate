"""Transactional orchestration for professional task exchange and analysis.

The ordinary store command path owns revisions, conflicts and receipts. This
module prepares typed changes inside that transaction; it does not write around
the command journal or interpret imported filesystem paths.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import stat

from . import schema as S
from .errors import SchemaInvalid, WorkspaceError
from .professional_schema import MAX_EVIDENCE_BYTES

# Leave room for the bounded command identity/revision envelope when a file
# produced by this exporter is returned through the1MiB HTTP command route.
EXPORT_DOCUMENT_LIMIT = 1024 * 1024 - 4096
PORTABLE_EVIDENCE_LIMIT = MAX_EVIDENCE_BYTES


def _artifact_descriptor(data):
    digest = hashlib.sha256(data).hexdigest()
    return {'id': S.make_id('art', digest), 'relative_path': 'objects/' + digest,
            'sha256': digest, 'size_bytes': str(len(data))}


def _verified_artifact_bytes(store, descriptor, *, limit, portable=False, include_data=True):
    """Verify a local object before reading a bounded export payload.

    Registration is not proof that the bytes still exist or that an opened
    workspace's object directory has not since been replaced with a link.
    """
    from .store import _safe

    if descriptor['relative_path'] != 'objects/' + descriptor['sha256']:
        raise WorkspaceError('RECOVERY_REQUIRED', 'Professional artifact path is invalid')
    path = store.root / descriptor['relative_path']
    _safe(path)
    expected_size = int(descriptor['size_bytes'])
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise WorkspaceError('RECOVERY_REQUIRED', 'Professional artifact bytes are missing or corrupt')
        if expected_size > limit:
            if portable:
                raise SchemaInvalid('PROFESSIONAL_LIMIT_EXCEEDED',
                                    'One evidence sample exceeds the portable export budget')
            raise WorkspaceError('RECOVERY_REQUIRED', 'Professional source archive exceeds its supported bound')
        data = bytearray() if include_data else None
        digest, read_size = hashlib.sha256(), 0
        with path.open('rb') as incoming:
            # A local admitted media object can be an unembedded evidence
            # reference. Verify it in bounded chunks without allocating its
            # entire payload, and read at most one byte beyond its descriptor.
            while read_size <= expected_size:
                chunk = incoming.read(min(1024 * 1024, expected_size - read_size + 1))
                if not chunk:
                    break
                read_size += len(chunk)
                digest.update(chunk)
                if data is not None:
                    data.extend(chunk)
    except OSError as exc:
        raise WorkspaceError('RECOVERY_REQUIRED', 'Professional artifact bytes are unavailable') from exc
    if read_size != expected_size or digest.hexdigest() != descriptor['sha256']:
        raise WorkspaceError('RECOVERY_REQUIRED', 'Professional artifact bytes are corrupt')
    return bytes(data) if data is not None else None


def prepare_task(store, current, operation, payload, *, command_id):
    from . import professional_schema as PS
    from .professional_export import import_payload

    task = copy.deepcopy(current)
    pending_artifacts = []
    media = store.get_media(task['media_id'])
    template = next(t for t in store.list_templates()
                    if (t['id'], t['version']) == (task['template_id'], task['template_version']))
    if operation == 'import_professional':
        if set(payload) != {'document'}:
            raise SchemaInvalid('SCHEMA_INVALID', 'Professional import requires one document')
        imported = import_payload(payload['document'], media, task['id'])
        # Preflight portable object identities before publishing any bytes.
        # A transaction can roll back registration but not immutable files.
        declared_artifacts = {}
        embedded_artifacts = set()
        for artifact in payload['document']['artifacts']:
            expected = {key: artifact[key] for key in ('id', 'relative_path', 'sha256', 'size_bytes')}
            declared_artifacts[artifact['id']] = expected
            if 'data_base64' not in artifact:
                continue
            data = base64.b64decode(artifact['data_base64'], validate=True)
            if _artifact_descriptor(data) != expected:
                raise SchemaInvalid('UNSUPPORTED_PROFESSIONAL_EVIDENCE',
                                    'Portable evidence identity differs from its bytes')
            embedded_artifacts.add(artifact['id'])
            pending_artifacts.append((expected, data))
        task['annotations'] = imported['annotations']
        task['extensions'] = imported['extensions']
        task['legacy_document_metadata'] = imported['legacy_document_metadata']
        archive = task['extensions']['professional']['import_source']
        source_document = archive.pop('document')
        # Keep the original exchange document exactly as JSON in immutable
        # project storage. Embedding it in every later task/export recursively
        # doubles the file on repeated round trips.
        source_data = S.canonical_json(source_document)
        archive['document_artifact'] = _artifact_descriptor(source_data)
        pending_artifacts.append((archive['document_artifact'], source_data))
        from .frames import validate_anchor
        model_run_ids = {run['id'] for run in task['extensions']['professional'].get('model_runs', [])}
        verified_local = set()
        evidence_owners = list(task['annotations'])
        evidence_owners.extend(task['extensions']['professional'].get('analysis_reports', []))
        for owner in evidence_owners:
            for ident in owner.get('evidence_ids', []):
                if ident in embedded_artifacts:
                    continue
                row = store.conn.execute('SELECT id,rel_path,sha256,size_bytes FROM artifacts WHERE id=?',
                                         (ident,)).fetchone()
                local = ({'id': row[0], 'relative_path': row[1], 'sha256': row[2],
                          'size_bytes': str(row[3])} if row is not None else None)
                if ident not in declared_artifacts or local != declared_artifacts[ident]:
                    raise SchemaInvalid('UNSUPPORTED_PROFESSIONAL_EVIDENCE',
                                        'Professional evidence is not embedded or matching locally admitted bytes')
                if ident not in verified_local:
                    _verified_artifact_bytes(store, local, limit=int(local['size_bytes']), include_data=False)
                    verified_local.add(ident)
        for annotation in task['annotations']:
            available = embedded_artifacts | set(annotation['evidence_ids'])
            PS.validate_annotation(annotation, media, template, artifact_ids=available,
                                   model_run_ids=model_run_ids)
            for key in ('start', 'end', 'at'):
                if 'anchor' in annotation.get(key, {}):
                    validate_anchor(store, media, annotation[key])
    elif operation == 'import_vision_candidates':
        candidates = payload.get('candidates')
        source_job_id = payload.get('source_job_id')
        model_runs = payload.get('model_runs')
        if (set(payload) != {'candidates', 'source_job_id', 'model_runs'}
                or not isinstance(candidates, list)
                or not 1 <= len(candidates) <= 4096 or not isinstance(source_job_id, str)
                or not isinstance(model_runs, list)):
            raise SchemaInvalid('SCHEMA_INVALID', 'Vision import requires a bounded candidate list and source job')
        S.require_id(source_job_id)
        from . import semantic_schema as SS
        if len(model_runs) > SS.MAX_MODEL_RUNS:
            raise SchemaInvalid('SCHEMA_INVALID', 'Vision import declares too many model runs')
        checked_runs = []
        run_ids = set()
        for run in model_runs:
            checked = SS.validate_model_run(run)
            if checked['id'] in run_ids:
                raise SchemaInvalid('SCHEMA_INVALID', 'Vision import declares a duplicate model run')
            run_ids.add(checked['id'])
            checked_runs.append(checked)
        professional = task.setdefault('extensions', {}).setdefault('professional', {})
        if not isinstance(professional, dict):
            raise WorkspaceError('RECOVERY_REQUIRED', 'Stored professional extension is invalid')
        declared = professional.setdefault('model_runs', [])
        if not isinstance(declared, list):
            raise WorkspaceError('RECOVERY_REQUIRED', 'Stored model runs are invalid')
        for run in checked_runs:
            existing = next((item for item in declared if item.get('id') == run['id']), None)
            if existing is not None and S.canonical_json(existing) != S.canonical_json(run):
                raise SchemaInvalid('SCHEMA_INVALID', 'Model run identities are immutable')
            if existing is None:
                declared.append(run)
        declared_ids = {item.get('id') for item in declared
                        if isinstance(item, dict) and isinstance(item.get('id'), str)}
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise SchemaInvalid('SCHEMA_INVALID', 'Vision candidate must be an object')
            if candidate['id'] in seen:
                raise SchemaInvalid('SCHEMA_INVALID', 'Vision candidate list contains duplicate identities')
            seen.add(candidate['id'])
            provenance = candidate.get('provenance')
            if (not isinstance(provenance, dict) or provenance.get('origin') != 'model_candidate'
                    or provenance.get('job_id') != source_job_id):
                raise SchemaInvalid('SCHEMA_INVALID',
                                    'Vision candidate must carry model_candidate provenance from its source job')
            declared = {row[0] for row in store.conn.execute("SELECT id FROM artifacts")}
            PS.validate_annotation(candidate, media, template, artifact_ids=declared,
                                   model_run_ids=declared_ids)
            task['annotations'].append(candidate)
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
    for expected, data in pending_artifacts:
        if store._put_artifact(data) != expected:
            raise WorkspaceError('RECOVERY_REQUIRED', 'Portable artifact publication changed its identity')
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


def apply_vision_candidates(store, task_id, *, command_id, expected_revision, expected_sha256,
                            candidates, source_job_id, model_runs=None, request_body=None):
    return store.apply_task_command(
        command_id=command_id, task_id=task_id, expected_revision=expected_revision,
        expected_sha256=expected_sha256, operation='import_vision_candidates',
        payload={'candidates': candidates, 'source_job_id': source_job_id,
                 'model_runs': list(model_runs or [])},
        request_body=request_body)['response']


def export_professional(store, task_id, *, evidence_mode='embedded'):
    from .professional_export import build_document
    from .professional_schema import MAX_DOCUMENT_BYTES

    if evidence_mode not in {'embedded', 'references'}:
        raise SchemaInvalid('UNSUPPORTED_EXPORT_OPTION', 'Evidence mode must be embedded or references')
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
            descriptor = {'id': row[0], 'relative_path': row[1], 'sha256': row[2], 'size_bytes': str(row[3])}
            if evidence_mode == 'references':
                _verified_artifact_bytes(store, descriptor, limit=int(descriptor['size_bytes']),
                                         include_data=False)
            artifacts.append(descriptor)
        known = {a['id'] for a in artifacts}
        for ident, descriptor in sorted(source_artifacts.items()):
            row = store.conn.execute('SELECT id,rel_path,sha256,size_bytes FROM artifacts WHERE id=?', (ident,)).fetchone()
            actual = ({'id': row[0], 'relative_path': row[1], 'sha256': row[2], 'size_bytes': str(row[3])}
                      if row is not None else None)
            if descriptor != actual:
                raise WorkspaceError('RECOVERY_REQUIRED', 'Professional source archive is missing or inconsistent')
            _verified_artifact_bytes(store, actual, limit=MAX_DOCUMENT_BYTES, include_data=False)
            if ident not in known:
                artifacts.append(actual)
                known.add(ident)
        evidence_owners = list(task['annotations'])
        evidence_owners.extend(task.get('extensions', {}).get('professional', {}).get('analysis_reports', []))
        for owner in evidence_owners:
            for ident in owner.get('evidence_ids', []):
                if ident in known:
                    continue
                row = store.conn.execute('SELECT id,rel_path,sha256,size_bytes FROM artifacts WHERE id=?', (ident,)).fetchone()
                if row is None:
                    raise WorkspaceError('RECOVERY_REQUIRED', 'Annotation evidence artifact is missing')
                descriptor = {'id': row[0], 'relative_path': row[1], 'sha256': row[2],
                              'size_bytes': str(row[3])}
                if evidence_mode == 'references':
                    _verified_artifact_bytes(store, descriptor, limit=int(descriptor['size_bytes']),
                                             include_data=False)
                    artifacts.append(descriptor)
                else:
                    data = _verified_artifact_bytes(store, descriptor,
                                                    limit=PORTABLE_EVIDENCE_LIMIT, portable=True)
                    artifacts.append({**descriptor, 'data_base64': base64.b64encode(data).decode('ascii')})
                known.add(ident)
        document = build_document(task, media, template, history=history, artifacts=artifacts)
        data = S.canonical_json(document)
        if len(data) > EXPORT_DOCUMENT_LIMIT:
            raise SchemaInvalid('PROFESSIONAL_LIMIT_EXCEEDED',
                                'Professional export exceeds its round-trip budget; keep the full project/backup')
        losses = 'Complete current task; audit index only, not historical revision bodies or archived source bytes. No media bytes. Preserve the full project/backup for history. Existing admitted source required for import.'
        if evidence_mode == 'references':
            losses += ' Evidence bytes are external object references. Evidence import requires matching locally admitted object hashes from the complete project/backup.'
        return {'data': data, 'content_type': 'application/json; charset=utf-8',
                'filename': f'TraceAnnotate-{task_id}-r{task["revision"]}.professional.json',
                'binding': {'task_id': task_id, 'revision': task['revision'],
                            'content_sha256': task['content_sha256']},
                'evidence_mode': evidence_mode, 'losses': losses}
