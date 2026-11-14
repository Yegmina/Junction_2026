"""Bounded positive-recovery SFT/GRPO search; no held-out data enters training."""
import argparse
import gc
import json
import os
from pathlib import Path


def run(spec):
    from aerorozum.common import save, utcnow, data_root
    from aerorozum.training import train
    from aerorozum.stream_training import evaluate_adapter, reward_signal_diagnostics
    import torch
    torch.set_num_threads(2)
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=True)
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE=spec['namespace'],
                      AEROROZUM_SFT_LEARNING_RATE=str(spec['learning_rate']),
                      AEROROZUM_DEVICE_MAP='balanced')
    if spec.get('warm_start'):
        os.environ['AEROROZUM_WARMSTART_ADAPTER'] = spec['warm_start']
    else:
        os.environ.pop('AEROROZUM_WARMSTART_ADAPTER', None)
    def stage(value):
        save(output/'status.json', dict(status='running', stage=value, at=utcnow(), **spec))
    results = {}
    try:
        stage('positive_recovery_sft')
        results['sft'] = train(spec['manifest'], 'stream_sft', 17, spec['sft_steps'])
        gc.collect(); torch.cuda.empty_cache()
        os.environ['AEROROZUM_SFT_CHECKPOINT'] = str(data_root()/results['sft']['adapter'])
        stage('dense_iou_grpo')
        results['grpo'] = train(spec['manifest'], 'stream_grpo', 17, spec['grpo_steps'])
        gc.collect(); torch.cuda.empty_cache()
        results['reward_signal'] = reward_signal_diagnostics((data_root()/results['grpo']['adapter']).parent/'metrics.jsonl')
        for name in ('sft', 'grpo'):
            stage('development_evaluation_'+name)
            results[name+'_evaluation'] = evaluate_adapter(data_root()/results[name]['adapter'], spec['validation'], output/'evaluation'/name,
                os.environ['AEROROZUM_LOCAL_MODEL_PATH'], seed=17)
            gc.collect(); torch.cuda.empty_cache()
        save(output/'result.json', dict(status='completed', at=utcnow(), specification=spec,
            results=results, frontier_superiority_verified=False, sealed_tests_accessed=False))
        save(output/'status.json', dict(status='completed', at=utcnow(), **spec))
    except Exception as error:
        save(output/'status.json', dict(status='failed', error=str(error)[:2000], at=utcnow(), **spec))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('specification'); args=parser.parse_args()
    run(json.loads(Path(args.specification).read_text()))
