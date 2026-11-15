"""Prospective paired GRPO replay comparison after HIGH-teacher evaluations.

Both arms start from the same frozen SFT adapter and retain every source row.
The only treatment is fourfold positive replay (80/20 effective class entries)
versus the balanced source. The existing zero-variance guard remains unchanged.
"""
from datetime import datetime,timedelta,timezone
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts';NIGHT=ART/'overnight-20261115T000942'


def helper():
    spec=importlib.util.spec_from_file_location('frozen_recipe',ROOT/'postsubmission/replicate_visual_recipe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare_source(parent_source):
    module=helper();parent_source=Path(parent_source);source=parent_source.with_name(parent_source.name+'-class-replay-v1')
    implementation=Path(__file__).resolve().parents[1]/'aerorozum/replay_curriculum.py'
    with (ROOT/'postsubmission/class-replay-source-v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not source.exists():
            shutil.copytree(parent_source,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            path=source/'aerorozum/training.py';code=path.read_text();needle="    memory_probe=os.environ.get('AEROROZUM_MEMORY_PROBE')=='1'"
            if code.count(needle)!=1:raise ValueError('Unexpected frozen training preamble')
            code=code.replace(needle,"    if stage=='stream_grpo':\n        from .replay_curriculum import repeat_positives\n        rows,protocol['class_replay']=repeat_positives(rows,int(os.environ.get('AEROROZUM_GRPO_POSITIVE_REPEATS','1')))\n"+needle)
            needle="    completed=output/'summary.json'"
            if code.count(needle)!=1:raise ValueError('Unexpected checkpoint identity guard')
            code=code.replace(needle,"    if prior.exists() and json.loads(prior.read_text()).get('class_replay')!=protocol.get('class_replay'):\n        raise ValueError('Replay weighting differs from the recorded checkpoint protocol')\n"+needle)
            compile(code,str(path),'exec');path.write_text(code);shutil.copyfile(implementation,source/'aerorozum/replay_curriculum.py')
            module.save(source/'class-replay-source-receipt.json',dict(parent_source=str(parent_source),
                parent_training_sha256=module.sha(parent_source/'aerorozum/training.py'),training_sha256=module.sha(path),
                replay_sha256=module.sha(implementation),all_existing_admission_and_gradient_guards_unchanged=True,
                zero_variance_patience_unchanged=4,source_examples_and_labels_unchanged=True))
        receipt=json.loads((source/'class-replay-source-receipt.json').read_text())
        if module.sha(source/'aerorozum/training.py')!=receipt['training_sha256'] or module.sha(source/'aerorozum/replay_curriculum.py')!=receipt['replay_sha256']:
            raise ValueError('Frozen replay source changed')
    return source


def train_phase(spec):
    from aerorozum.common import save
    from aerorozum.training import train
    from aerorozum.stream_training import reward_signal_diagnostics
    for key in ['AEROROZUM_GRPO_CONTINUATION_RESULT','AEROROZUM_WARMSTART_ADAPTER','AEROROZUM_TRAINING_AUGMENTATION']:
        os.environ.pop(key,None)
    os.environ.update(AEROROZUM_SFT_CHECKPOINT=spec['parent_sft_adapter'],AEROROZUM_TRAINING_NAMESPACE=spec['namespace'],
        AEROROZUM_LORA_SCOPE=spec['lora_scope'],AEROROZUM_DEVICE_MAP='balanced',AEROROZUM_MICROBATCH='1',AEROROZUM_GRAD_ACCUMULATION='4',
        AEROROZUM_GRPO_POSITIVE_REPEATS=str(spec['positive_repeats']))
    result=train(spec['manifest'],'stream_grpo',spec['seed'],96)
    trace=ART/Path(result['adapter']).parent/'metrics.jsonl'
    save(Path(spec['output'])/'training-result.json',dict(training=result,reward_signal=reward_signal_diagnostics(trace)))


def work(seed,gpu):
    module=helper();parent_path=ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}/initial/result.json'
    parent=json.loads(parent_path.read_text());parent_spec=parent['specification'];parent_adapter=ART/parent['results']['sft']['adapter']
    if parent.get('status')!='completed' or parent['results']['sft']['seed']!=seed:raise ValueError('Completed same-seed SFT parent required')
    for path,expected in parent_spec['input_hashes'].items():
        if module.sha(path)!=expected:raise ValueError('A frozen HIGH-teacher input changed')
    source=prepare_source(parent_spec['source_initial']);root=ART/f'postsubmission-high-teacher-replay-v1/seed-{seed}'
    if root.exists():raise ValueError('Preserve an existing paired replay outcome')
    lease=dict(id='high-teacher-replay-'+str(seed),gpu_ids=[gpu],expires_at=(datetime.now(timezone.utc)+timedelta(hours=4.5)).isoformat(),purpose='Paired 50/50 versus 80/20 replay from the same SFT adapter; no auto promotion')
    module.lease(ROOT,lease)
    try:
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if gpu==0 or uuids[gpu] in used:raise ValueError('Preserve occupied GPUs')
        for arm,repeats in [('balanced',1),('positive-replay',4)]:
            out=root/arm;namespace=f'postsubmission-high-teacher-replay-v1/seed-{seed}/'+arm
            spec=dict(parent_spec,kind='paired_class_replay',namespace=namespace,output=str(out),lease=lease,
                source_initial=str(source),source_continuation=str(source),sft_steps=0,grpo_steps=96,positive_repeats=repeats,
                parent_sft_adapter=str(parent_adapter),parent_adapter_sha256=module.sha(parent_adapter/'adapter_model.safetensors'),
                same_frozen_sft_parent_for_both_arms=True,matching_maximum_updates=96,actual_compute_matching_claimed=False,
                four_zero_variance_group_stop_unchanged=True,automatic_promotion=False,test_accessed=False)
            files=[Path(__file__),parent_path,Path(spec['manifest']),Path(spec['validation']),parent_adapter/'adapter_model.safetensors',
                   source/'aerorozum/training.py',source/'aerorozum/replay_curriculum.py',source/'configs/research.json']
            spec['input_hashes']={str(p):module.sha(p) for p in files};module.save(out/'specification.json',spec)
            module.save(root/'status.json',dict(status='running',seed=seed,stage=arm))
            module.save(out/'status.json',dict(status='running',seed=seed,stage='grpo_up_to_96'))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(ART),
                AEROROZUM_LOCAL_MODEL_PATH=str(ROOT/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
                HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            subprocess.run([str(ROOT/'venv/bin/python'),'-u',str(Path(__file__).resolve()),'--train-spec',str(out/'specification.json')],env=env,check=True)
            result=json.loads((out/'training-result.json').read_text());adapter=ART/result['training']['adapter']
            module.save(out/'status.json',dict(status='running',seed=seed,stage='development_evaluation'))
            subprocess.run([str(ROOT/'venv/bin/python'),'-u','-m','aerorozum.stream_training','--evaluate-only','--adapter',str(adapter),
                '--validation',spec['validation'],'--output',str(out/'evaluation'),'--seed',str(seed)],env=env,check=True)
            result['grpo_evaluation']=json.loads((out/'evaluation/summary.json').read_text())
            module.save(out/'result.json',dict(status='completed',specification=spec,results=result,
                interpretation='Paired development comparison of training-example weighting. Actual update counts and compute can differ because the stop guard is unchanged.'))
            module.save(out/'status.json',dict(status='completed',seed=seed,test_accessed=False))
        module.save(root/'status.json',dict(status='completed',seed=seed,test_accessed=False,automatic_promotion=False))
    except Exception as error:
        module.save(root/'status.json',dict(status='failed',seed=seed,error=str(error)[:1500]));raise
    finally:module.lease(ROOT,lease,remove=True)


def queue():
    module=helper();protocol=NIGHT/'paired-class-replay-protocol.json'
    if not protocol.exists():module.save(protocol,dict(status='predeclared_before_high_teacher_results',seeds=[17,43,91],
        parent='Each seed exact full HIGH-teacher SFT adapter',arms=dict(balanced=1,positive_replay=4),
        every_source_example_retained=True,maximum_grpo_updates_per_arm=96,zero_variance_patience=4,
        fixed_development_images=120,test_accessed=False,automatic_promotion=False,
        hypothesis='An explicit positive replay curriculum may reduce runs of zero-variance model-zero groups without weakening the four-group stop rule. All outcomes, false-alarm rates and actual update counts are retained.'))
    pending={17:1,43:2,91:3};deadline=datetime.fromisoformat('2026-11-16T06:00:00+00:00')
    while pending and datetime.now(timezone.utc)<deadline:
        for seed,gpu in list(pending.items()):
            parent=ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}/status.json'
            evaluation=ART/f'postsubmission-high-teacher-snapshots-v1/seed-{seed}/status.json'
            if not parent.exists() or not evaluation.exists():continue
            if json.loads(parent.read_text()).get('status')=='failed':pending.pop(seed);continue
            if json.loads(parent.read_text()).get('status')!='completed' or json.loads(evaluation.read_text()).get('status') not in ('completed','failed'):continue
            if subprocess.run(['systemctl','is-active','aerorozum-high-teacher-snapshots-'+str(seed)],capture_output=True,text=True).stdout.strip() in ('active','activating'):continue
            log=NIGHT/f'paired-class-replay-{seed}.log'
            subprocess.run(['systemd-run','--unit=aerorozum-high-teacher-replay-'+str(seed),'--property=RuntimeMaxSec=14400','--property=CPUQuota=300%','--property=MemoryMax=64G',
                '--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),str(ROOT/'venv/bin/python'),'-u',str(Path(__file__).resolve()),'--seed',str(seed),'--gpu',str(gpu)],check=True,timeout=30)
            pending.pop(seed)
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int);parser.add_argument('--gpu',type=int);parser.add_argument('--train-spec');args=parser.parse_args()
    if args.train_spec:train_phase(json.loads(Path(args.train_spec).read_text()))
    elif args.seed is None:queue()
    else:work(args.seed,args.gpu)
