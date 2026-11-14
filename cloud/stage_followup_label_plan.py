"""Stage checked follow-up frames and resume the existing, frozen ER2 worker.

This does not change its prompt, provider code, quota settings, or archive policy.
Prior paid identities, including invalid responses, are excluded from new work.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess

MODEL = 'gemini-robotics-er-2-preview'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(row):
    return row['sha256'], str(row['source']['generation'])


def pending_rows(document, audit, records):
    if document.get('policy', {}).get('version') != 'airborne-only-v2-primary-state':
        raise ValueError('Corrected airborne-only qualification is required')
    paid = {identity(r) for r in records if r.get('model') == MODEL}
    bindings = {(f['id'], f['sha256'], str(f['source']['generation'])): f
                for f in audit['frames'] if f['eligibility']['status'] == 'eligible'}
    pending, seen = [], set()
    for row in document['images']:
        key = (row['id'], row['sha256'], str(row['source']['generation']))
        decision = row.get('eligibility', {})
        if key not in bindings or bindings[key]['eligibility'] != decision:
            raise ValueError('Frame is not bound to the qualification audit')
        receipt = row.get('receipt_timestamp_ns')
        if (not isinstance(receipt, int) or isinstance(receipt, bool) or receipt <= 0
                or row.get('airborne_eligible') is not True or decision.get('landed_state') != 2
                or not 0 <= decision.get('state_age_s', float('inf')) <= 2
                or receipt != decision.get('thermal_receipt_ns')):
            raise ValueError('Fresh original in-air receipt is required')
        if row['id'] in seen:
            raise ValueError('Duplicate frame identity')
        seen.add(row['id'])
        if identity(row) not in paid:
            pending.append(row)
    return pending


def save(path, document, account=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.new')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    if account:
        os.chown(temporary, account.pw_uid, account.pw_gid)
        temporary.chmod(0o600)
    temporary.replace(path)


def assert_idle(home):
    pid = subprocess.check_output(['systemctl', 'show', 'aerorozum-airborne-labeler',
                                   '--property=MainPID', '--value'], text=True, timeout=10).strip()
    if pid != '0':
        raise ValueError('Existing label worker is live; do not change its plan')
    state = json.loads((home / 'artifacts/pipeline/label-worker.json').read_text())
    if state.get('paid_calls_settled') is not True:
        raise ValueError('Previous paid requests are not confirmed settled')


def stage(args):
    followup, home = args.followup_root.resolve(), args.labeler_home.resolve()
    assert_idle(home)
    result = json.loads((followup / 'result.json').read_text())
    if result.get('status') not in ('prepared', 'prepared_with_quarantines'):
        raise ValueError('Follow-up preparation is not terminal')
    raw_audit = (followup / 'airborne-eligibility.json').read_bytes()
    audit = json.loads(raw_audit)
    document = json.loads((followup / 'airborne-eligible.json').read_text())
    if hashlib.sha256(raw_audit).hexdigest() != document['eligibility_manifest_sha256']:
        raise ValueError('Follow-up audit hash differs')
    records = [json.loads(line) for line in args.paid_index.read_text().splitlines() if line.strip()]
    pending = pending_rows(document, audit, records)
    if not pending:
        print(json.dumps({'status': 'no_new_paid_identities', 'eligible_frames': len(document['images'])}))
        return
    account = pwd.getpwnam('aerorozum-labeler')
    root = home / 'artifacts'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    relative = Path('pipeline/inputs/airborne-snapshots') / ('followup-' + stamp)
    snapshot = root / relative
    snapshot.mkdir(parents=True, exist_ok=False)
    old_plan = json.loads((home / 'state/launch.json').read_text())
    old_full = json.loads(Path(old_plan['manifest_path']).read_text())
    for sid, split in old_full.get('session_splits', {}).items():
        if document['session_splits'].get(sid) != split:
            raise ValueError('A frozen session split changed')
    staged = []
    for row in document['images']:
        row = copy.deepcopy(row)
        source = (followup / row['image']).resolve()
        if not source.is_relative_to(followup) or sha(source) != row['sha256']:
            raise ValueError('Prepared image escaped its source tree or changed')
        image = Path('labelling/followup-thermal-v1/frames') / (row['sha256'] + '.jpg')
        target = root / image
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != row['sha256']:
                raise ValueError('Existing staged image differs')
        else:
            shutil.copyfile(source, target)
            target.chmod(0o644)
        row['image'] = image.as_posix()
        staged.append(row)
    (snapshot / 'airborne-eligibility.json').write_bytes(raw_audit)
    full = {**document, 'images': staged,
            'eligibility_manifest': (relative / 'airborne-eligibility.json').as_posix()}
    save(snapshot / 'airborne-eligible.json', full)
    pending_ids = {r['id'] for r in pending}
    subset = {**full, 'images': [r for r in staged if r['id'] in pending_ids], 'subset': 'pending_only',
              'full_manifest': (relative / 'airborne-eligible.json').as_posix(),
              'full_manifest_sha256': sha(snapshot / 'airborne-eligible.json')}
    save(snapshot / 'airborne-pending.json', subset)
    plan = copy.deepcopy(old_plan)
    plan.pop('input_archive', None)
    plan.update(not_before_utc=datetime.now(timezone.utc).isoformat(),
                manifest_path=str(snapshot / 'airborne-pending.json'),
                manifest_sha256=sha(snapshot / 'airborne-pending.json'),
                eligibility_manifest_path=str(snapshot / 'airborne-eligibility.json'),
                input_staging={'method': 'exact_local_vm_copy', 'source_audit_sha256': sha(snapshot / 'airborne-eligibility.json'),
                               'prior_paid_index_sha256': sha(args.paid_index), 'eligible_frames': len(staged),
                               'pending_frames': len(pending), 'prior_paid_frames_excluded': len(staged) - len(pending)})
    candidate = home / 'state' / ('launch-followup-' + stamp + '.json')
    save(candidate, plan, account)
    environment = dict(os.environ, PYTHONPATH=str(home / 'source'), AEROROZUM_DATA_ROOT=str(root),
                       AEROROZUM_LABEL_PROMPT_VERSION=plan['label_prompt_version'])
    subprocess.run(['runuser', '-u', 'aerorozum-labeler', '--', '/opt/aerorozum/venv/bin/python',
                    str(home / 'source/cloud/airborne_worker.py'), '--plan', str(candidate), '--validate-only'],
                   env=environment, check=True)
    receipt = {'status': 'validated_no_paid_calls', 'at': datetime.now(timezone.utc).isoformat(),
               'snapshot': str(snapshot), 'manifest_sha256': plan['manifest_sha256'],
               'prompt_version': plan['label_prompt_version'], 'prompt_sha256': plan['prompt_sha256'],
               'source_code_changed': False, 'source_versions': result['source_versions'],
               'source_quarantines': result['quarantined_sources'], **plan['input_staging']}
    if args.start:
        assert_idle(home)
        old = home / 'state' / ('launch-before-followup-' + stamp + '.json')
        shutil.copy2(home / 'state/launch.json', old)
        save(home / 'state/launch.json', plan, account)
        subprocess.run(['systemctl', 'start', 'aerorozum-airborne-labeler'], check=True, timeout=30)
        receipt['status'] = 'validated_label_worker_started'
    save(snapshot / 'staging-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup-root', type=Path, required=True)
    parser.add_argument('--labeler-home', type=Path, default=Path('/opt/aerorozum/airborne-labeler'))
    parser.add_argument('--paid-index', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    stage(parser.parse_args())
