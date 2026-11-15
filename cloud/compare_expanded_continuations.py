"""Predeclared paired update controls after expanded parents finish.

Same parent, seed, images and optimizer-update count: +96 SFT versus +96 GRPO.
Generation makes their compute unequal; this is explicitly not compute matching.
"""
import argparse
from datetime import datetime,timedelta,timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts'


def helper():
    spec=importlib.util.spec_from_file_location('frozen_recipe',ROOT/'postsubmission/replicate_visual_recipe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def continuation_source(module):
    source=ROOT/'postsubmission/grpo-expanded-source-v1'
    if not source.exists():
        original=ROOT/'harbor/grpo-only-source-v1'
        shutil.copytree(original,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        path=source/'aerorozum/training.py';code=path.read_text()
        if code.count('max_completion_length=384')!=1:raise ValueError('Detector completion limit changed')
        code=code.replace('max_completion_length=384','max_completion_length=512')
        start=code.index('                squared=0.;count=0;visual_squared=0.')
        end=code.index("                if lora_scope=='language_visual_projection'",start)
        code=code[:start]+'''                from .gradient_monitor import statistics
                observed=statistics(model.named_parameters())
                squared=observed['squared'];count=observed['count'];visual_squared=observed['visual_squared']
'''+code[end:]
        path.write_text(code)
        checked=ROOT/'postsubmission/optimized-training-source-v1/aerorozum/gradient_monitor.py'
        receipt=json.loads((ART/'overnight-20261115T000942/gradient-monitor-benchmark.json').read_text())
        if module.sha(checked)!=receipt['monitor_source_sha256']:raise ValueError('Qualified monitor differs')
        shutil.copyfile(checked,source/'aerorozum/gradient_monitor.py')
        module.save(source/'source-receipt.json',dict(original=str(original),original_training_sha256=module.sha(original/'aerorozum/training.py'),
            changes=['Detector completion cap 512','Qualified read-only gradient monitor'],original_optimizer_unchanged=True))
    return source


def run(seed,gpu):
    module=helper();parent_path=ART/f'postsubmission-expanded-visual-v1/seed-{seed}/initial/result.json'
    parent=json.loads(parent_path.read_text())
    if parent['status']!='completed':raise ValueError('A completed expanded parent is required')
    parent_spec=parent['specification'];parent_adapter=ART/parent['results']['grpo']['adapter']
    if parent['results']['grpo']['seed']!=seed:raise ValueError('Parent seed differs')
    for path,expected in parent_spec['input_hashes'].items():
        if module.sha(path)!=expected:raise ValueError('A frozen expanded input changed: '+path)
    out=ART/f'postsubmission-expanded-controls-v1/seed-{seed}'
    if out.exists():raise ValueError('Preserve an existing comparison rather than replacing it')
    import fcntl
    with (ROOT/'postsubmission/grpo-expanded-source-v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=continuation_source(module)
    lease=dict(id='expanded-controls-'+str(seed),gpu_ids=[gpu],expires_at=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat(),purpose='Paired SFT/GRPO continuation controls; no automatic promotion')
    module.lease(ROOT,lease)
    try:
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if gpu==0 or uuids[gpu] in used:raise ValueError('Preserve an occupied GPU')
        module.save(out/'status.json',dict(status='running',stage='matched_sft_updates'))
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),AEROROZUM_DATA_ROOT=str(ART),
            AEROROZUM_LOCAL_MODEL_PATH=str(ROOT/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
            HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        spec=dict(parent_spec,seed=seed,kind='matched_sft_updates',sft_steps=96,grpo_steps=0,continuation_steps=0,
            learning_rate=1e-6,output=str(out/'sft-control'),namespace=f'postsubmission-expanded-controls-v1/seed-{seed}/sft-control',
            parent_adapter=str(parent_adapter),lease=lease,compute_matching_claimed=False)
        files=[Path(__file__),parent_path,parent_adapter/'adapter_model.safetensors',Path(spec['manifest']),Path(spec['validation'])]
        spec['input_hashes']={str(p):module.sha(p) for p in files}
        module.save(out/'sft-control/specification.json',spec)
        subprocess.run([sys.executable,'-u',str(ROOT/'postsubmission/replicate_visual_recipe.py'),'phase',str(out/'sft-control/specification.json'),'--phase','initial'],env=dict(environment,PYTHONPATH=spec['source_initial']),check=True)
        if parent['results'].get('reward_signal',{}).get('task_signal_verified') is not True:
            module.save(out/'grpo-not-admitted.json',dict(reason='Parent lacked relative task reward variation',guard_unchanged=True))
        else:
            module.save(out/'status.json',dict(status='running',stage='grpo_96'))
            spec=dict(spec,kind='expanded_grpo_continuation',output=str(out/'grpo'),namespace=f'postsubmission-expanded-controls-v1/seed-{seed}/grpo',
                sft_steps=0,grpo_steps=0,continuation_steps=96,source_continuation=str(source))
            files.extend([source/'aerorozum/training.py',source/'aerorozum/gradient_monitor.py',source/'configs/research.json'])
            spec['input_hashes']={str(p):module.sha(p) for p in files}
            module.save(out/'grpo/initial/result.json',parent)
            module.save(out/'grpo/specification.json',spec)
            subprocess.run([sys.executable,'-u',str(ROOT/'postsubmission/replicate_visual_recipe.py'),'phase',str(out/'grpo/specification.json'),'--phase','continuation'],env=dict(environment,PYTHONPATH=str(source)),check=True)
        module.save(out/'status.json',dict(status='completed',seed=seed,test_accessed=False,automatic_promotion=False))
    except Exception as error:
        if out.exists():module.save(out/'status.json',dict(status='failed',seed=seed,error=str(error)[:1500]))
        raise
    finally:module.lease(ROOT,lease,remove=True)


def queue():
    pending={17:1,43:2,91:3};deadline=datetime.fromisoformat('2026-11-15T04:00:00+00:00')
    while pending and datetime.now(timezone.utc)<deadline:
        for seed,gpu in list(pending.items()):
            parent=ART/f'postsubmission-expanded-visual-v1/seed-{seed}/status.json'
            if not parent.exists():continue
            status=json.loads(parent.read_text()).get('status')
            if status=='failed':pending.pop(seed);continue
            if status!='completed':continue
            if (ART/f'postsubmission-expanded-controls-v1/seed-{seed}').exists():pending.pop(seed);continue
            out=ART/f'postsubmission-expanded-controls-v1/seed-{seed}';out.mkdir(parents=True,exist_ok=True)
            # The worker creates its own directory only after GPU admission.
            out.rmdir()
            command=['systemd-run','--unit=aerorozum-expanded-controls-'+str(seed),
                '--property=RuntimeMaxSec=9000','--property=CPUQuota=300%','--property=MemoryMax=64G',
                '--property=StandardOutput=append:'+str(ART/f'overnight-20261115T000942/expanded-controls-{seed}.log'),
                '--property=StandardError=append:'+str(ART/f'overnight-20261115T000942/expanded-controls-{seed}.log'),
                str(ROOT/'venv/bin/python'),'-u',str(Path(__file__).resolve()),'--seed',str(seed),'--gpu',str(gpu)]
            subprocess.run(command,check=True,timeout=30);pending.pop(seed)
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[17,43,91]);parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.seed is None:queue()
    else:run(args.seed,args.gpu)
