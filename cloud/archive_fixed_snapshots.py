"""Back up completed read-only snapshots while their parent training continues."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path('/opt/aerorozum')
ART = ROOT / 'artifacts'
NIGHT = ART / 'overnight-20261115T000942'
BUCKET = 'aerorozum-thermal-rl-avid-relic-509408-e4'
PREFIX = 'research/overnight-20261115/fixed-sft-snapshots'
END = datetime.fromisoformat('2026-11-16T16:30:00+00:00')


def archive(folder, seed, store):
    from aerorozum.common import save, utcnow
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['rng_streams_unchanged'] is True
    assert receipt['parameter_and_gradient_versions_unchanged'] is True
    step = receipt['optimizer_step']
    spec_path = ART / f'postsubmission-high-teacher-visual-v1/seed-{seed}/specification.json'
    spec = json.loads(spec_path.read_text())
    assert step in spec['fixed_sft_snapshot_steps']
    assert (folder / 'adapter/adapter_model.safetensors').is_file()
    assert (folder / 'adapter/adapter_config.json').is_file()
    files = {}
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        assert not path.is_symlink()
        raw = path.read_bytes()
        files[str(path.relative_to(folder))] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                                    md5=base64.b64encode(hashlib.md5(raw).digest()).decode())
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    prefix = PREFIX + f'/seed-{seed}/step-{step}/' + identity
    subprocess.run(['/snap/bin/gcloud', 'storage', 'rsync', '-r', str(folder),
                    'gs://' + BUCKET + '/' + prefix + '/', '--quiet'], check=True, timeout=600)
    generations = {}
    for relative, metadata in files.items():
        name = prefix + '/' + relative
        request = urllib.request.Request('https://storage.googleapis.com/storage/v1/b/' + BUCKET +
                                         '/o/' + urllib.parse.quote(name, safe=''), headers=store._headers())
        with urllib.request.urlopen(request, timeout=60) as response:
            remote = json.load(response)
        assert int(remote['size']) == metadata['size'] and remote['md5Hash'] == metadata['md5']
        assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == metadata['sha256']
        generations[relative] = remote['generation']
    result = dict(at=utcnow(), seed=seed, optimizer_step=step, status='server_digest_verified',
                  snapshot_identity_sha256=identity, files=files, object_generations=generations,
                  specification_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  uri='gs://' + BUCKET + '/' + prefix + '/', no_local_deletion=True,
                  parent_training_unchanged=True, automatic_promotion=False)
    store.put(identity, result, 'progress')
    save(NIGHT / 'fixed-snapshot-archives' / f'seed-{seed}-step-{step}.json', result)
    return result


def main():
    from cloud.airborne_worker import metadata_record_store
    store = metadata_record_store(BUCKET, prefix=PREFIX + '/manifests')
    while datetime.now(timezone.utc) < END:
        for seed in (17, 43, 91):
            base = ART / 'training' / f'postsubmission-high-teacher-visual-v1/seed-{seed}'
            for receipt in sorted(base.glob('**/fixed-sft-snapshots/step-*/receipt.json')):
                step = json.loads(receipt.read_text())['optimizer_step']
                done = NIGHT / 'fixed-snapshot-archives' / f'seed-{seed}-step-{step}.json'
                if done.exists():
                    continue
                try:
                    result = archive(receipt.parent, seed, store)
                    print(json.dumps(dict(seed=seed, step=step, uri=result['uri'])), flush=True)
                except Exception as error:
                    print(type(error).__name__ + ': ' + str(error)[:1000], flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
