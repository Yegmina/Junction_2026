"""Drain already accepted work and finalize the user-requested research wrap-up."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from cloud.wrap_up_cloud import ROOT, OUT, BUCKET, PREFIX, active, archive

PUBLIC = ROOT / 'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'
NIGHT = OUT.parent


def publish(value):
    temporary = PUBLIC / '.wrap-up.json.next'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o644)
    temporary.replace(PUBLIC / 'wrap-up.json')


def native_units():
    result = subprocess.run(['systemctl', 'list-units', '--state=running', '--no-legend',
                             'aerorozum-overnight-gemini-*', 'aerorozum-overnight-gemma-native16k-*'],
                            capture_output=True, text=True, check=True)
    return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]


def main():
    from aerorozum.common import save, utcnow
    from cloud.airborne_worker import metadata_record_store
    from cloud.archive_overnight import sync
    deadline = time.monotonic() + 5400
    training = json.loads((OUT / 'training-wrap-up.json').read_text())
    assert all(row['status'] == 'checkpointed_on_user_request' for row in training['results'].values())
    while time.monotonic() < deadline:
        remaining = native_units()
        review_active = active('aerorozum-post-label-reviews-v2')
        state = dict(at=utcnow(), status='draining_existing_work', new_experiments_disabled=True,
                     active_native_trials=remaining, review_archiver_active=review_active,
                     saved_training_steps={seed: row['saved_optimizer_updates'] for seed, row in training['results'].items()},
                     full_research_completed=False, live_demo_unchanged=True,
                     summary='GPU training is checkpointed and cloud-verified. New experiments are stopped. '
                     'The final review archive and already-running mission are draining; the existing CPU physics solve can finish.')
        publish(state)
        if not remaining and not review_active: break
        time.sleep(15)
    else: raise TimeoutError('Existing work did not drain; preserve all active services and checkpoints for inspection')
    report = json.loads((NIGHT / 'post-label-background-tiles-v1/report.json').read_text())
    assert report['status'] == 'completed' and report['completed'] == 6000
    review_archive = json.loads((NIGHT / 'post-label-background-tiles-archive.json').read_text())
    assert review_archive['status'] == 'synced' and not review_archive['failures']
    # Reconcile the retained lossless crop inputs, without issuing inference.
    pixels = NIGHT / 'post-label-background-tiles-v1/derived-inputs'
    subprocess.run(['/snap/bin/gcloud', 'storage', 'rsync', '-r', str(pixels),
                    'gs://' + BUCKET + '/research/overnight-20261115/review-pixels/background-tiles/', '--quiet'],
                   check=True, timeout=1800, capture_output=True)
    # Only trials finishing during wrap-up need a final reconciliation after
    # their wrapper has written instance-check and other last log records.
    cutoff = json.loads((OUT / 'dispatch-stopped.json').read_text())['at']
    reconciled = []
    for path in (ROOT / 'harbor/jobs').glob('overnight-*/result.json'):
        result = json.loads(path.read_text())
        if result.get('finished_at') and result['finished_at'][:19] >= cutoff[:19]:
            sync(path.parent, 'native-missions/' + path.parent.name)
            reconciled.append(path.parent.name)
    proxy_units = subprocess.check_output(['systemctl', 'list-units', '--state=running', '--no-legend',
                                           'aerorozum-native16k-proxy-*'], text=True).splitlines()
    for row in proxy_units:
        subprocess.run(['systemctl', 'stop', row.split()[0]], check=True, timeout=60)
    # All fixed snapshots are included in the verified frozen training archive.
    subprocess.run(['systemctl', 'stop', 'aerorozum-fixed-snapshot-archive-v2'], check=False, capture_output=True, timeout=60)
    physics_active = active('aerorozum-physics-3d-object-h4-domain2')
    physics_freezer = subprocess.run(['systemctl', 'show', 'aerorozum-physics-3d-object-h4-domain2',
                                     '-p', 'FreezerState', '--value'], capture_output=True, text=True).stdout.strip()
    result = dict(at=utcnow(), status='research_checkpointed_and_archived', new_experiments_disabled=True,
                  saved_training_steps=state['saved_training_steps'], final_review=report,
                  raw_review_archive_versions=review_archive['archived_versions'],
                  final_native_logs_reconciled=reconciled, full_research_completed=False,
                  existing_physics_solver_running=physics_active and physics_freezer == 'running',
                  existing_physics_solver_freezer_state=physics_freezer,
                  live_demo_unchanged=True, vm_shutdown_requested=False, automatic_model_promotion=False,
                  checkpoint_restore_training_test_performed=False)
    save(OUT / 'wrap-up-result.json', result)
    receipt = archive(OUT, 'final-manifests')
    result['wrap_up_archive'] = dict(uri=receipt['uri'], server_digest_verified=receipt['server_digest_verified'])
    save(OUT / 'wrap-up-result-with-archive.json', result)
    store = metadata_record_store(BUCKET, prefix=PREFIX + '/final-result')
    store.put(hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(), result, 'progress')
    result['summary'] = ('GPU experiments are stopped at verified resumable checkpoints. Labels, reviews and final logs '
                         'are archived. The live demo stays online. Full physics qualification remains incomplete. '
                         + ('The existing CPU solve can finish without launching another case.'
                            if result['existing_physics_solver_running'] else 'The physics solve is not currently advancing.'))
    publish(result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        from aerorozum.common import save, utcnow, redacted_error
        record = dict(at=utcnow(), status='wrap_up_needs_attention', error=redacted_error(error),
                      new_experiments_disabled=True, full_research_completed=False,
                      summary='Training checkpoints are preserved. Final archival needs attention; no new experiments are launching.')
        save(OUT / 'finalizer-error.json', record)
        publish(record)
        raise
