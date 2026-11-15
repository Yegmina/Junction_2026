"""Launch two bounded continuations of the measured visual-projection adapter.

Runs only on explicitly idle GPUs 1 and 3. Development selection remains
separate from frozen test zones and native mission-policy evaluation.
"""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root=Path('/opt/aerorozum');artifacts=root/'artifacts'
    source=root/'harbor/visual-training-source'
    parent=artifacts/'training/submission-visual-projection-v2/updates-480/google_gemma-4-12B-it/stream_grpo/seed-17/adapter'
    summary=json.loads((parent.parent/'summary.json').read_text())
    assert summary['status']=='completed' and summary['lora_scope']=='language_visual_projection'
    assert summary['nonzero_gradient_checks']>0
    parent_hash=hashlib.sha256((parent/'adapter_model.safetensors').read_bytes()).hexdigest()
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    gpu_ids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    for gpu in (1,3):
        if gpu_ids[gpu] in query:raise RuntimeError(f'GPU {gpu} has a live compute process; refusing to interrupt it')
    for gpu,steps,lr in [(1,240,1e-5),(3,480,3e-5)]:
        namespace=f'submission-visual-continuation-v1/updates-{steps}'
        out=artifacts/namespace;out.mkdir(parents=True,exist_ok=True)
        specification=out/'specification.json'
        if specification.exists():raise RuntimeError('Continuation already recorded; inspect its authoritative unit before retrying')
        spec=dict(namespace=namespace,manifest=str(artifacts/'submission-refinement-v1/train.json'),
            validation=str(artifacts/'pipeline/validation.json'),output=str(out),learning_rate=lr,
            sft_steps=steps,grpo_steps=96,warm_start=str(parent),warm_start_sha256=parent_hash,
            lora_scope='language_visual_projection',scope='Development-only perception continuation; no frozen test access')
        specification.write_text(json.dumps(spec,indent=2)+'\n')
        args=['systemd-run',f'--unit=aerorozum-visual-continuation-v1-gpu{gpu}',
            '--property=RuntimeMaxSec=10800','--property=CPUQuota=300%','--property=MemoryMax=64G',
            f'--working-directory={source}',f'--property=StandardOutput=append:{out}/worker.log',
            f'--property=StandardError=append:{out}/worker.log']
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(artifacts),
            AEROROZUM_LOCAL_MODEL_PATH=str(root/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
            AEROROZUM_LORA_SCOPE='language_visual_projection',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        args.extend('--setenv='+key+'='+value for key,value in env.items())
        args.extend([str(root/'venv/bin/python'),'-u',str(root/'deployment/recovery/refine_perception.py'),str(specification)])
        subprocess.run(args,check=True)
        print(json.dumps(dict(gpu=gpu,namespace=namespace,parent_sha256=parent_hash)),flush=True)


if __name__=='__main__':main()
