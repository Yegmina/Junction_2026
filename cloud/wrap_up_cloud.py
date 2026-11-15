"""Checkpoint and archive active research on user request; keep the demo online.

No model source, training target, verifier or physical-solver configuration is
changed. An interrupted run is never labeled completed or given a fake score.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/opt/aerorozum')
ART = ROOT / 'artifacts'
OUT = ART / 'overnight-20261115T000942/wrap-up-v1'
BUCKET = 'aerorozum-thermal-rl-avid-relic-509408-e4'
PREFIX = 'research/overnight-20261115/wrap-up-v1'


def command(*args):
    return subprocess.run(list(args), capture_output=True, text=True, check=True, timeout=120)


def active(unit):
    return subprocess.run(['systemctl', 'is-active', unit], capture_output=True, text=True).stdout.strip() in ('active', 'activating')


def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with path.open('rb') as stream:
        while data := stream.read(2 * 1024 * 1024):
            sha.update(data); md5.update(data); size += len(data)
    return dict(size=size, sha256=sha.hexdigest(), md5=base64.b64encode(md5.digest()).decode())


def archive(folder, relative):
    from cloud.airborne_worker import metadata_record_store
    store = metadata_record_store(BUCKET, prefix=PREFIX + '/manifests')
    files = {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    prefix = PREFIX + '/' + relative
    command('/snap/bin/gcloud', 'storage', 'rsync', '-r', str(folder), 'gs://' + BUCKET + '/' + prefix + '/', '--quiet')
    remote = {}; page = None
    while True:
        query = dict(prefix=prefix + '/', maxResults=1000, fields='items(name,size,md5Hash,generation),nextPageToken')
        if page: query['pageToken'] = page
        request = urllib.request.Request('https://storage.googleapis.com/storage/v1/b/' + BUCKET +
                                         '/o?' + urllib.parse.urlencode(query), headers=store._headers())
        with urllib.request.urlopen(request, timeout=60) as response: data = json.load(response)
        remote.update({row['name'][len(prefix)+1:]: row for row in data.get('items', [])})
        page = data.get('nextPageToken')
        if not page: break
    generations = {}
    for name, metadata in files.items():
        obj = remote[name]
        assert int(obj['size']) == metadata['size'] and obj['md5Hash'] == metadata['md5'], name
        assert digest(folder / name) == metadata, 'Source changed during archival: ' + name
        generations[name] = obj['generation']
    receipt = dict(uri='gs://' + BUCKET + '/' + prefix + '/', files=files,
                   object_generations=generations, server_digest_verified=True, no_local_deletion=True)
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    store.put(key, receipt, 'progress')
    return receipt


def valid_checkpoint(path):
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                'adapter_model.safetensors', 'adapter_config.json', 'training_args.bin']
    if not all((path / name).is_file() for name in required): return False
    state = json.loads((path / 'trainer_state.json').read_text())
    if state['global_step'] != int(path.name.split('-')[-1]): return False
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin']:
        with zipfile.ZipFile(path / name) as container:
            if container.testzip() is not None: return False
    with (path / 'adapter_model.safetensors').open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little')
        assert 0 < length < 10 * 1024 * 1024
        header = json.loads(stream.read(length))
    ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
    return bool(ends) and 8 + length + max(ends) == (path / 'adapter_model.safetensors').stat().st_size


def checkpoint_training(seed):
    from aerorozum.common import save, utcnow
    unit = 'aerorozum-high-teacher-visual-' + str(seed)
    parent = ART / f'postsubmission-high-teacher-visual-v1/seed-{seed}'
    spec = json.loads((parent / 'specification.json').read_text())
    training = ART / 'training' / spec['namespace']
    done = OUT / f'training-{seed}.json'
    if done.exists(): return json.loads(done.read_text())
    if not active(unit):
        result = dict(seed=seed, status='already_inactive', original_status=json.loads((parent/'status.json').read_text()))
        save(done, result); return result
    states = list(training.glob('**/checkpoint-*/trainer_state.json'))
    baseline = max(int(p.parent.name.split('-')[-1]) for p in states)
    target = baseline + 25
    save(OUT / f'training-{seed}-progress.json', dict(at=utcnow(), status='waiting_for_next_regular_checkpoint', seed=seed, target_step=target))
    deadline = time.monotonic() + 600
    selected = None
    while time.monotonic() < deadline and active(unit):
        candidates = sorted(training.glob('**/checkpoint-*/trainer_state.json'),
                            key=lambda p: int(p.parent.name.split('-')[-1]), reverse=True)
        if candidates and int(candidates[0].parent.name.split('-')[-1]) >= target:
            command('systemctl', 'freeze', unit)
            assert command('systemctl', 'show', unit, '-p', 'FreezerState', '--value').stdout.strip() == 'frozen'
            try:
                if valid_checkpoint(candidates[0].parent): selected = candidates[0].parent; break
            except (OSError, ValueError, zipfile.BadZipFile): pass
            command('systemctl', 'thaw', unit)
        time.sleep(.5)
    if selected is None: raise RuntimeError('No new complete checkpoint; preserve active run for inspection')
    # The unit remains frozen during upload and verification, including its
    # child trainer. No file can be rotated by save_total_limit in this window.
    save(OUT / f'training-{seed}-progress.json', dict(at=utcnow(), status='frozen_at_saved_checkpoint_archiving', seed=seed, checkpoint=str(selected)))
    receipt = archive(training, f'training/seed-{seed}')
    step = int(selected.name.split('-')[-1])
    result = dict(at=utcnow(), seed=seed, status='checkpointed_on_user_request', checkpoint=str(selected),
                  saved_optimizer_updates=step, planned_optimizer_updates=spec['sft_steps'],
                  optimizer_scheduler_rng_saved=True, training_archive=receipt,
                  full_training_completed=False, grpo_completed=False, automatic_promotion=False,
                  source_and_training_configuration_unchanged=True)
    save(OUT / f'training-{seed}-verified-before-stop.json', result)
    command('systemctl', 'stop', unit)
    assert not active(unit)
    save(parent / 'wrap-up.json', result)
    for path in [parent / 'status.json', parent / 'initial/status.json']:
        if path.exists(): save(OUT / f'seed-{seed}-{path.parent.name}-original-status.json', json.loads(path.read_text()))
        save(path, dict(at=utcnow(), status=result['status'], seed=seed, saved_optimizer_updates=step,
                        planned_optimizer_updates=spec['sft_steps'], full_training_completed=False, automatic_promotion=False))
    module_spec = importlib.util.spec_from_file_location('frozen_recipe_wrap', ROOT / 'postsubmission/replicate_visual_recipe.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.lease(ROOT, spec['lease'], remove=True)
    result['experiment_archive'] = archive(parent, f'experiments/seed-{seed}')
    save(done, result)
    print(json.dumps(dict(seed=seed, status=result['status'], saved_optimizer_updates=step)), flush=True)
    return result


def main():
    from aerorozum.common import save, utcnow
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {seed: pool.submit(checkpoint_training, seed) for seed in (17, 43, 91)}
        results = {}
        for seed, future in futures.items():
            try: results[str(seed)] = future.result()
            except Exception as error:
                results[str(seed)] = dict(status='wrap_up_needs_attention', error=str(error)[:1500])
                print(json.dumps(dict(seed=seed, **results[str(seed)])), flush=True)
    save(OUT / 'training-wrap-up.json', dict(at=utcnow(), status='checkpointing_finished', results=results,
                                           full_research_completed=False, live_demo_unchanged=True))


if __name__ == '__main__': main()
