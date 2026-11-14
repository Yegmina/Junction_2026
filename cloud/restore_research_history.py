"""Restore research metadata after moving the public UI to a separate data root.

Run with the recovered SSD archive directory, original worker artifact root,
and deployed artifact root. Original source records remain unchanged. No jobs
are started and no model weights or credentials are copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.restore-tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def restore(source, worker, target):
    source, worker, target = map(Path, (source, worker, target))
    assert source.resolve() != target.resolve()
    snapshot = read(source/'status.json')
    if not snapshot.get('training'):
        raise ValueError('Recovered research snapshot has no training metadata')
    stamp = datetime.now(timezone.utc).isoformat()
    # Only missing files are imported; current deployment data always survives.
    copied = []
    for name in ('campaigns', 'validation'):
        for path in (source/name).rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = target/relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                copied.append(str(relative))
    for origin, historical in ((source, True), (worker, False)):
        for path in (origin/'jobs').glob('*.json'):
            job = read(path)
            destination = target/'jobs'/path.name
            current = read(destination)
            if current and current.get('updated_at', '') >= job.get('updated_at', ''):
                continue
            if historical and job.get('status') in ('running', 'queued', 'starting'):
                job['recorded_status'] = job['status']
                job['status'] = 'historical_snapshot'
                job['detail'] = ('Historical record; last reported '+job['recorded_status']+
                                 ' at '+job.get('updated_at', 'unknown time')+
                                 '. Current labelling activity is reported in Pipeline. '+
                                 job.get('detail', ''))
            job['history_source'] = str(path)
            write(destination, job)
    # The worker observer mirrors training/status.json. Restore its missing
    # source too, so the next telemetry tick cannot replace it with {}.
    training = read(worker/'training/status.json') or read(source/'training/status.json')
    if not read(worker/'training/status.json'):
        write(worker/'training/status.json', training)
    if not read(target/'training/status.json'):
        write(target/'training/status.json', training)
    snapshot['training'] = training
    snapshot.pop('jobs', None)  # Job files remain the ledger, including worker jobs.
    current = read(target/'status.json')
    merged = {**snapshot, **current}
    if not current.get('training'):
        merged['training'] = training
    merged['history_restored_at'] = stamp
    merged.pop('jobs', None)
    write(target/'status.json', merged)
    receipt = dict(restored_at=stamp, recovered_snapshot_sha256=hashlib.sha256(
        (source/'status.json').read_bytes()).hexdigest(), copied_files=copied,
        jobs=len(list((target/'jobs').glob('*.json'))),
        campaign_states=len(list((target/'campaigns').rglob('state.json'))),
        training_status=training.get('status'), source_records_modified=False)
    write(target/'recovery/research-history.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('worker')
    parser.add_argument('target')
    args = parser.parse_args()
    result = restore(args.source, args.worker, args.target)
    print(json.dumps({k:v for k,v in result.items() if k != 'copied_files'}))
