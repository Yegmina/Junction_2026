"""Frozen visual-recipe replications and a matched-update SFT control.

Training/evaluation run in the original source snapshots, on development data
only. There is no model promotion or access to the opened held-out test.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def recipe(parent, seed, control=False):
    specification = parent['specification']
    if parent.get('status') != 'completed' or seed not in (17, 43, 91):
        raise ValueError('A completed parent and a declared seed are required')
    if (specification.get('sft_steps'), specification.get('grpo_steps'), specification.get('learning_rate')) != (480, 48, 3e-5):
        raise ValueError('The established 480/48 visual recipe differs')
    for name in ['sft', 'grpo']:
        summary = parent['results'][name]
        if (summary.get('lora_scope'), summary.get('microbatch'), summary.get('gradient_accumulation')) != ('language_visual_projection', 1, 4):
            raise ValueError('Parent scope or effective-batch topology differs')
    if control and seed != 17:
        raise ValueError('The matched SFT control must use its seed-17 parent')
    return dict(seed=seed, kind='matched_sft_updates' if control else 'fixed_recipe_replication',
                sft_steps=96 if control else 480, grpo_steps=0 if control else 48,
                continuation_steps=0 if control else 96, learning_rate=1e-6 if control else 3e-5,
                microbatch=1, gradient_accumulation=4, lora_scope='language_visual_projection',
                test_access=False, automatic_promotion=False,
                compute_matching_claimed=False, selection_rule='Report all declared runs; no test-based selection')


def lease(root, entry, remove=False):
    import fcntl
    path = Path(root) / 'artifacts/pipeline/controls.json'
    with path.with_suffix('.lease-lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        controls = json.loads(path.read_text())
        leases = [row for row in controls.get('external_gpu_leases', []) if row['id'] != entry['id']]
        if not remove:
            leases.append(entry)
        controls['external_gpu_leases'] = leases
        save(path, controls)


def phase(spec, name):
    import gc
    import torch
    from aerorozum.common import data_root
    from aerorozum.training import train
    from aerorozum.stream_training import evaluate_adapter, reward_signal_diagnostics
    torch.set_num_threads(2)
    out = Path(spec['output']) / name
    out.mkdir(parents=True, exist_ok=True)
    seed = spec['seed']
    for key in ['AEROROZUM_WARMSTART_ADAPTER', 'AEROROZUM_GRPO_CONTINUATION_RESULT',
                'AEROROZUM_SFT_CHECKPOINT', 'AEROROZUM_TRAINING_AUGMENTATION']:
        os.environ.pop(key, None)
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE=spec['namespace'] + '/' + name,
                      AEROROZUM_LORA_SCOPE=spec['lora_scope'], AEROROZUM_DEVICE_MAP='balanced',
                      AEROROZUM_SFT_LEARNING_RATE=str(spec['learning_rate']),
                      AEROROZUM_MICROBATCH='1', AEROROZUM_GRAD_ACCUMULATION='4')
    results = {}
    def status(stage):
        save(out / 'status.json', dict(status='running', stage=stage, seed=seed))
    if name == 'continuation':
        os.environ['AEROROZUM_GRPO_CONTINUATION_RESULT'] = str(Path(spec['output']) / 'initial/result.json')
        status('grpo_96')
        results['grpo'] = train(spec['manifest'], 'stream_grpo', seed, spec['continuation_steps'])
    else:
        if spec['kind'] == 'matched_sft_updates':
            os.environ['AEROROZUM_WARMSTART_ADAPTER'] = spec['parent_adapter']
        status('sft_' + str(spec['sft_steps']))
        results['sft'] = train(spec['manifest'], 'stream_sft', seed, spec['sft_steps'])
        gc.collect(); torch.cuda.empty_cache()
        if spec['grpo_steps']:
            os.environ['AEROROZUM_SFT_CHECKPOINT'] = str(data_root() / results['sft']['adapter'])
            status('grpo_48')
            results['grpo'] = train(spec['manifest'], 'stream_grpo', seed, spec['grpo_steps'])
    gc.collect(); torch.cuda.empty_cache()
    if 'grpo' in results:
        results['reward_signal'] = reward_signal_diagnostics((data_root() / results['grpo']['adapter']).parent / 'metrics.jsonl')
    for key in ['sft', 'grpo']:
        if key not in results:
            continue
        status('development_evaluation_' + key)
        results[key + '_evaluation'] = evaluate_adapter(data_root() / results[key]['adapter'], spec['validation'],
            out / 'evaluation' / key, os.environ['AEROROZUM_LOCAL_MODEL_PATH'], seed=seed)
        gc.collect(); torch.cuda.empty_cache()
    save(out / 'result.json', dict(status='completed', specification=spec, results=results,
                                  sealed_tests_accessed=False, automatic_promotion=False))
    save(out / 'status.json', dict(status='completed', seed=seed))


def worker(spec):
    out = Path(spec['output'])
    try:
        for file, expected in spec['input_hashes'].items():
            if sha(file) != expected:
                raise ValueError('A frozen input or source snapshot changed: ' + file)
        for name in ['initial', 'continuation']:
            if name == 'continuation':
                if not spec['continuation_steps']:
                    break
                initial = json.loads((out / 'initial/result.json').read_text())
                if initial['results'].get('reward_signal', {}).get('task_signal_verified') is not True:
                    save(out / 'continuation-not-admitted.json', {'reason': 'Parent has no verified task reward variation', 'guard_unchanged': True})
                    break
            save(out / 'status.json', dict(status='running', stage=name, seed=spec['seed']))
            environment = dict(os.environ, PYTHONPATH=spec['source_initial' if name == 'initial' else 'source_continuation'])
            subprocess.run([sys.executable, '-u', __file__, 'phase', str(out / 'specification.json'), '--phase', name],
                           env=environment, check=True)
        save(out / 'status.json', dict(status='completed', seed=spec['seed'], test_access=False))
    except Exception as error:
        save(out / 'status.json', dict(status='failed', error=str(error)[:2000], seed=spec['seed']))
        raise
    finally:
        lease(spec['root'], spec['lease'], remove=True)


def launch(root):
    root = Path(root).resolve()
    parent_file = root / 'artifacts/submission-visual-projection-v2/updates-480/result.json'
    parent = json.loads(parent_file.read_text())
    study = root / 'artifacts/postsubmission-visual-replication-v1'
    if study.exists():
        raise ValueError('Study already exists; inspect its units before retrying')
    source_initial = root / 'harbor/visual-training-source'
    source_continuation = root / 'harbor/grpo-only-source-v1'
    files = [Path(__file__).resolve(), parent_file, Path(parent['specification']['manifest']), Path(parent['specification']['validation'])]
    for source in [source_initial, source_continuation]:
        files.extend(source / name for name in ['aerorozum/training.py', 'aerorozum/stream_training.py', 'configs/research.json'])
    parent_adapter = root / 'artifacts' / parent['results']['grpo']['adapter']
    files.append(parent_adapter / 'adapter_model.safetensors')
    hashes = {str(file): sha(file) for file in files}
    # Reserve the GPUs before checking again, so the streaming scheduler yields.
    plans = []
    for gpu, seed, control in [(1, 17, True), (2, 43, False), (3, 91, False)]:
        spec = recipe(parent, seed, control)
        name = 'sft-control-17' if control else 'seed-' + str(seed)
        out = study / name
        spec.update(root=str(root), namespace='postsubmission-visual-replication-v1/' + name,
                    manifest=parent['specification']['manifest'], validation=parent['specification']['validation'],
                    output=str(out), source_initial=str(source_initial), source_continuation=str(source_continuation),
                    parent_adapter=str(parent_adapter), input_hashes=hashes,
                    lease={'id': 'visual-replication-' + name, 'gpu_ids': [gpu],
                           'expires_at': (datetime.now(timezone.utc) + timedelta(hours=3, minutes=15)).isoformat(),
                           'purpose': 'Declared fixed-recipe replication or matched-update control'})
        plans.append((gpu, spec))
    for _, spec in plans:
        lease(root, spec['lease'])
    started = set()
    try:
        import time
        time.sleep(1)
        state = json.loads((root / 'artifacts/pipeline/coordinator.json').read_text())
        if any(j.get('status') == 'running' for key in ['jobs', 'evaluation_jobs'] for j in state.get(key, [])):
            raise ValueError('Streaming workers are live; preserve them before scheduling this study')
        uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
        used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
        if any(uuids[gpu] in used for gpu, _ in plans):
            raise ValueError('A declared GPU still has a compute process')
        for gpu, spec in plans:
            out = Path(spec['output']);out.mkdir(parents=True)
            save(out / 'specification.json', spec)
            environment = dict(CUDA_VISIBLE_DEVICES=str(gpu), AEROROZUM_DATA_ROOT=str(root / 'artifacts'),
                AEROROZUM_LOCAL_MODEL_PATH=str(root / 'models/gemma-4-12B-it'), AEROROZUM_GPU_MEMORY_FRACTION='.88',
                HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            command = ['systemd-run', '--unit=aerorozum-visual-replication-' + str(spec['seed']),
                '--property=RuntimeMaxSec=10800', '--property=CPUQuota=300%', '--property=MemoryMax=64G',
                '--property=StandardOutput=append:' + str(out / 'worker.log'),
                '--property=StandardError=append:' + str(out / 'worker.log')]
            command.extend('--setenv=' + key + '=' + value for key, value in environment.items())
            command.extend([str(root / 'venv/bin/python'), '-u', __file__, 'worker', str(out / 'specification.json')])
            subprocess.run(command, check=True, timeout=30)
            started.add(spec['lease']['id'])
            print(json.dumps({'seed': spec['seed'], 'gpu': gpu, 'kind': spec['kind'], 'output': spec['output']}), flush=True)
        save(study / 'protocol.json', {'status': 'launched', 'plans': [s for _, s in plans],
             'submission_artifacts_changed': False, 'sealed_tests_accessed': False,
             'comparison': 'Replicate the fixed successful recipe across seeds; report matched optimizer steps separately from compute'})
    finally:
        for _, spec in plans:
            if spec['lease']['id'] not in started:
                lease(root, spec['lease'], remove=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'worker', 'phase'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--phase', choices=['initial', 'continuation'])
    args = parser.parse_args()
    if args.action == 'launch':
        launch(args.path)
    elif args.action == 'worker':
        worker(json.loads(args.path.read_text()))
    else:
        phase(json.loads(args.path.read_text()), args.phase)
