"""Evaluate fixed early SFT adapters after each HIGH-teacher worker releases GPU.

All fixed snapshots are retained and use the existing 120-image development
protocol. There is no automatic deployment or new opened-test evaluation.
"""
from datetime import datetime,timedelta,timezone
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts'


def helper():
    spec=importlib.util.spec_from_file_location('frozen_recipe',ROOT/'postsubmission/replicate_visual_recipe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def work(seed,gpu):
    module=helper();parent=ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}'
    spec=json.loads((parent/'specification.json').read_text());out=ART/f'postsubmission-high-teacher-snapshots-v1/seed-{seed}'
    for path,expected in spec['input_hashes'].items():
        if module.sha(path)!=expected:raise ValueError('A frozen HIGH-teacher input changed: '+path)
    candidates=list((ART/'training'/spec['namespace']).rglob('fixed-sft-snapshots/step-*/adapter'))
    candidates.sort(key=lambda p:int(p.parent.name.removeprefix('step-')))
    expected=set(spec['fixed_sft_snapshot_steps']);actual={int(p.parent.name.removeprefix('step-')) for p in candidates}
    if not actual<=expected:raise ValueError('Unplanned early checkpoint')
    if not candidates:raise ValueError('No read-only snapshots were saved; preserve the parent failure')
    lease=dict(id='high-teacher-snapshots-'+str(seed),gpu_ids=[gpu],expires_at=(datetime.now(timezone.utc)+timedelta(hours=4.5)).isoformat(),purpose='Fixed early SFT checkpoint development evaluation; no automatic promotion')
    module.lease(ROOT,lease)
    try:
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if gpu==0 or uuids[gpu] in used:raise ValueError('The parent GPU is still occupied')
        module.save(out/'specification.json',dict(spec,kind='fixed_sft_checkpoint_evaluation',output=str(out),
            namespace=f'postsubmission-high-teacher-snapshots-v1/seed-{seed}',lease=lease,sft_steps=0,grpo_steps=0,
            checkpoint_adapters={str(p):module.sha(p/'adapter_model.safetensors') for p in candidates},
            test_accessed=False,automatic_promotion=False))
        results=[]
        for adapter in candidates:
            step=int(adapter.parent.name.removeprefix('step-'));folder=out/('step-'+str(step))
            module.save(out/'status.json',dict(status='running',stage='development_evaluation_step_'+str(step),step=step,seed=seed,test_accessed=False,automatic_promotion=False))
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=spec['source_initial'],
                AEROROZUM_DATA_ROOT=str(ART),AEROROZUM_LOCAL_MODEL_PATH=str(ROOT/'models/gemma-4-12B-it'),
                AEROROZUM_GPU_MEMORY_FRACTION='.88',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            command=[str(ROOT/'venv/bin/python'),'-u','-m','aerorozum.stream_training','--evaluate-only','--adapter',str(adapter),
                '--validation',spec['validation'],'--output',str(folder),'--seed',str(seed)]
            subprocess.run(command,check=True,env=environment)
            result=json.loads((folder/'summary.json').read_text());results.append(dict(optimizer_step=step,
                actual_epoch=json.loads((adapter.parent/'receipt.json').read_text())['actual_epoch'],
                adapter_sha256=module.sha(adapter/'adapter_model.safetensors'),evaluation=result))
        summary=dict(status='completed',seed=seed,expected_fixed_steps=sorted(expected),saved_fixed_steps=sorted(actual),
            all_fixed_steps_saved=actual==expected,results=results,development_manifest_sha256=module.sha(spec['validation']),
            primary_parent_evaluation_unchanged=True,test_accessed=False,automatic_promotion=False,
            interpretation='Additional fixed early checkpoints evaluated on repeatedly used development references; no independent field-accuracy claim.')
        module.save(out/'summary.json',summary);module.save(out/'status.json',dict(status='completed',seed=seed,test_accessed=False))
    except Exception as error:
        module.save(out/'status.json',dict(status='failed',seed=seed,error=str(error)[:1500]));raise
    finally:module.lease(ROOT,lease,remove=True)


def queue():
    pending={17:1,43:2,91:3};deadline=datetime.fromisoformat('2026-11-16T11:30:00+00:00')
    while pending and datetime.now(timezone.utc)<deadline:
        for seed,gpu in list(pending.items()):
            parent=ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}/status.json'
            if not parent.exists() or json.loads(parent.read_text()).get('status') not in ('completed','failed'):continue
            unit='aerorozum-high-teacher-visual-'+str(seed)
            active=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip()
            if active in ('active','activating'):continue
            out=ART/f'postsubmission-high-teacher-snapshots-v1/seed-{seed}'
            if out.exists():pending.pop(seed);continue
            log=ART/f'overnight-20261115T000942/snapshot-evaluation-{seed}.log'
            command=['systemd-run','--unit=aerorozum-high-teacher-snapshots-'+str(seed),'--property=RuntimeMaxSec=14400','--property=CPUQuota=300%','--property=MemoryMax=64G',
                '--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),str(ROOT/'venv/bin/python'),'-u',str(Path(__file__).resolve()),'--seed',str(seed),'--gpu',str(gpu)]
            subprocess.run(command,check=True,timeout=30);pending.pop(seed)
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.seed is None:queue()
    else:work(args.seed,args.gpu)
