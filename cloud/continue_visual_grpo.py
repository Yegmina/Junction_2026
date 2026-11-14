"""Bounded GRPO-only continuation; no new SFT or held-out test access."""
import argparse
import gc
import json
import os
from pathlib import Path


def run(spec):
    import torch
    from aerorozum.common import data_root,save,utcnow
    from aerorozum.training import train
    from aerorozum.stream_training import evaluate_adapter,reward_signal_diagnostics
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE=spec['namespace'],AEROROZUM_LORA_SCOPE='language_visual_projection',
        AEROROZUM_GRPO_CONTINUATION_RESULT=spec['parent_result'],AEROROZUM_DEVICE_MAP='balanced')
    os.environ.pop('AEROROZUM_TRAINING_AUGMENTATION',None)
    torch.set_num_threads(2)
    def status(state,**more):save(output/'status.json',dict(status=state,at=utcnow(),**spec,**more))
    try:
        status('running',stage='grpo_only_continuation')
        trained=train(spec['manifest'],'stream_grpo',17,spec['grpo_steps'])
        gc.collect();torch.cuda.empty_cache()
        signal=reward_signal_diagnostics((data_root()/trained['adapter']).parent/'metrics.jsonl')
        status('running',stage='development_evaluation_grpo')
        evaluation=evaluate_adapter(data_root()/trained['adapter'],spec['validation'],output/'evaluation/grpo',
            os.environ['AEROROZUM_LOCAL_MODEL_PATH'],seed=17)
        parent=json.loads(Path(spec['parent_result']).read_text())
        result=dict(status='completed',at=utcnow(),specification=spec,
            results=dict(grpo=trained,grpo_evaluation=evaluation,reward_signal=signal,
                         parent_evaluation=parent['results']['grpo_evaluation']),
            additional_sft_updates=0,optimizer_state_resumed=False,sealed_tests_accessed=False,
            frontier_superiority_verified=False)
        save(output/'result.json',result);status('completed')
    except Exception as error:
        status('failed',error=str(error)[:2000]);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('specification');args=parser.parse_args()
    run(json.loads(Path(args.specification).read_text()))
