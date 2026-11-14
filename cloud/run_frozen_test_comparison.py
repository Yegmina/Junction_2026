"""Open the held-out ZIP split once, after bounded development selection ends."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def active(unit):return subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0


def main():
    root=Path('/opt/aerorozum');data=root/'artifacts';source=Path(__file__).resolve().parents[1]
    from aerorozum.frozen_test import freeze_selection,prepare_manifest
    deadline=datetime.datetime(2026,11,14,22,22,tzinfo=datetime.timezone.utc)
    units=['aerorozum-grpo-only-v1-gpu2','aerorozum-visual-continuation-v1-gpu3']
    while any(active(unit) for unit in units) and datetime.datetime.now(datetime.timezone.utc)<deadline:time.sleep(10)
    output=data/'submission-frozen-test-v1';output.mkdir(exist_ok=True)
    selection=output/'selection.json';manifest=output/'manifest.json'
    candidates=[data/name/'result.json' for name in ['submission-visual-projection-v2/updates-120',
        'submission-visual-projection-v2/updates-480','submission-visual-continuation-v1/updates-240',
        'submission-visual-continuation-v1/updates-480','submission-grpo-continuation-v1/updates-96']]
    receipt=freeze_selection(data,candidates,selection)
    document=prepare_manifest(data,selection,data/'datasets/real/manifest.json',
        data/'submission-refinement-v1/train.json',data/'pipeline/validation.json',manifest)
    print(json.dumps(dict(selection_frozen=True,selected_development_f2=receipt['selected']['micro_f2'],test_images=len(document['images']))),flush=True)
    common=dict(PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(data),
        AEROROZUM_LOCAL_MODEL_PATH=str(root/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
        HF_HUB_OFFLINE='1',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
    runner=str(source/'cloud/evaluate_frozen_test.py');base=['--selection',str(selection),'--manifest',str(manifest),'--output',str(output/'evaluation')]
    python=str(root/'venv/bin/python');launched=[]
    for model in ['gemini-4-carbon','gemini-4-argon','gemini-robotics-er-2-preview']:
        unit='aerorozum-frozen-test-'+model
        env={**common,'AEROROZUM_GEMINI_KEY_FILE':'/etc/aerorozum/hackathon-key'}
        cmd=['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=1800','--property=CPUQuota=150%','--property=MemoryMax=5G']
        cmd+=['--setenv='+k+'='+v for k,v in env.items()]
        subprocess.run(cmd+[python,'-u',runner,*base,'--model',model],check=True);launched.append(unit)
    # Release no process: wait for one of the development-training GPUs to exit.
    gpu=None
    while gpu is None and datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,14,22,30,tzinfo=datetime.timezone.utc):
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        for index,unit in [(2,units[0]),(3,units[1])]:
            if not active(unit) and uuids[index] not in used:gpu=index;break
        if gpu is None:time.sleep(10)
    if gpu is None:raise RuntimeError('No free training GPU before cutoff; do not interrupt active work')
    cmd=['systemd-run','--unit=aerorozum-frozen-test-gemma','--property=RuntimeMaxSec=2100','--property=CPUQuota=300%','--property=MemoryMax=64G']
    cmd+=['--setenv='+k+'='+v for k,v in {**common,'CUDA_VISIBLE_DEVICES':str(gpu)}.items()]
    subprocess.run(cmd+[python,'-u',runner,*base],check=True);launched.append('aerorozum-frozen-test-gemma')
    while any(active(unit) for unit in launched):time.sleep(10)
    summaries=list((output/'evaluation').glob('*/summary.json'))
    print(json.dumps(dict(terminal=True,completed_summaries=len(summaries),expected=4)),flush=True)


if __name__=='__main__':main()
