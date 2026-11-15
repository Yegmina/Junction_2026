"""Release the terminal base-control server, then run a bounded GRPO-only arm."""
import json
from pathlib import Path
import subprocess
import time


def main():
    root=Path('/opt/aerorozum');h=root/'harbor';job='opencode-gemma-native-base-v2'
    while subprocess.run(['systemctl','is-active','--quiet','aerorozum-harbor-'+job]).returncode==0:time.sleep(10)
    results=list((h/'jobs'/job).glob('*/result.json'))
    if len(results)!=1:raise RuntimeError('Expected a terminal base-control result')
    result=json.loads(results[0].read_text())
    if not result.get('finished_at') or result.get('exception_info'):
        raise RuntimeError('Inspect the incomplete or errored base trial before releasing its GPU')
    if (h/'augmented-training-source').exists():raise RuntimeError('An augmentation experiment owns this GPU; do not interrupt it')
    source=h/'grpo-only-source-v1';assert (h/'grpo-only-preflight.json').is_file()
    for unit in ['aerorozum-gemma-native-base-v2-proxy','aerorozum-gemma-native-base-proxy',
                 'aerorozum-gemma-base-opencode-proxy','aerorozum-gemma-base-control']:
        subprocess.run(['systemctl','stop',unit],check=False)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if uuids[2] in used:raise RuntimeError('GPU 2 has another live process; do not overlap training')
    namespace='submission-grpo-continuation-v1/updates-96';out=root/'artifacts'/namespace;out.mkdir(parents=True,exist_ok=True)
    spec=dict(namespace=namespace,output=str(out),manifest=str(root/'artifacts/submission-refinement-v1/train.json'),
        validation=str(root/'artifacts/pipeline/validation.json'),grpo_steps=96,
        parent_result=str(root/'artifacts/submission-visual-projection-v2/updates-480/result.json'),
        additional_sft_updates=0,scope='GRPO-only development continuation with a fresh optimizer; no sealed-test access')
    file=out/'specification.json'
    if file.exists():raise RuntimeError('Continuation already recorded; inspect before retrying')
    file.write_text(json.dumps(spec,indent=2)+'\n')
    args=['systemd-run','--unit=aerorozum-grpo-only-v1-gpu2','--property=RuntimeMaxSec=5400',
        '--property=CPUQuota=300%','--property=MemoryMax=64G',f'--working-directory={source}',
        f'--property=StandardOutput=append:{out}/worker.log',f'--property=StandardError=append:{out}/worker.log']
    env=dict(CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(root/'artifacts'),
        AEROROZUM_LOCAL_MODEL_PATH=str(root/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
        HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    args.extend('--setenv='+k+'='+v for k,v in env.items())
    args.extend([str(root/'venv/bin/python'),'-u',str(source/'continue_visual_grpo.py'),str(file)])
    subprocess.run(args,check=True)


if __name__=='__main__':main()
