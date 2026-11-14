"""Back up paid response attempts independently of inference worker lifetimes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from .common import digest, redacted_error, save, utcnow


def archive_pass(folder, store, state_path, workers=8):
    """Archive immutable JSON versions, including failed/truncated paid attempts.

    This function never constructs an inference provider or changes worker files.
    A failed upload remains eligible on the next pass. Changed progress and result
    documents get new content-addressed versions rather than overwriting history.
    """
    folder = Path(folder).resolve()
    state_path = Path(state_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'records': {}}
    records = state['records']
    pending = []
    failures = []
    paths = set(folder.glob('*.json'))
    for name in ('results', 'paid-responses', 'provider/raw', 'provider/responses', 'provider/errors', 'failures', 'checkpoints', 'runs'):
        paths.update((folder / name).glob('*.json'))
    for path in sorted(paths):
        try:
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            relative = path.relative_to(folder).as_posix()
            key = digest([relative, sha])
            if key in records:
                continue
            original = json.loads(payload)
            # Raw provider/error records are preserved byte-for-byte inside the
            # original JSON value; no reconstructed reasoning or response text.
            pending.append((key, relative, sha, original))
        except (OSError, ValueError) as exc:
            failures.append({'path': str(path.relative_to(folder)), 'error': redacted_error(exc)})

    def upload(item):
        key, relative, sha, original = item
        wrapped = {'relative_path': relative, 'source_json_sha256': sha,
                   'archived_at': utcnow(), 'original': original}
        uri = store.put(key, wrapped, kind='logs')
        return key, {'path': relative, 'sha256': sha, 'uri': uri}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload, item): item for item in pending}
        for count, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                key, record = future.result()
                records[key] = record
            except Exception as exc:
                failures.append({'path': item[1], 'error': redacted_error(exc)})
            if count % 32 == 0:
                save(state_path, state)
    state.update(updated_at=utcnow(), archived_versions=len(records),
                 attempted=len(pending), failures=failures,
                 status='upload_pending' if failures else 'synced')
    # Save locally before the remote index; a network failure never loses the
    # completed upload ledger and never triggers fresh inference.
    save(state_path, state)
    try:
        index_key = digest({'records': records, 'failures': failures})
        state['index_uri'] = store.put(index_key, state, kind='progress')
    except Exception as exc:
        state['status'] = 'upload_pending'
        state['failures'].append({'path': 'archive-index', 'error': redacted_error(exc)})
    save(state_path, state)
    return state
