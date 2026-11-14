"""Queue a frozen-data augmentation experiment after the live base trial ends."""
import argparse,json
from pathlib import Path
import shutil
import subprocess
import time


def main(base_job):
    root=Path('/opt/aerorozum');harbor=root/'harbor'
    blocker='aerorozum-harbor-'+base_job
    while subprocess.run(['systemctl','is-active','--quiet',blocker]).returncode==0:time.sleep(10)
    results=list((harbor/'jobs'/base_job).glob('*/result.json'))
    if len(results)!=1:raise RuntimeError('Expected one terminal base-control trial')
    result=json.loads(results[0].read_text())
    if not result.get('finished_at') or result.get('exception_info'):
        raise RuntimeError('Inspect the incomplete or errored base trial before releasing its GPU')
    source=harbor/'augmented-training-source'
    if source.exists():raise RuntimeError('Refuse to modify a pre-existing training source snapshot')
    shutil.copytree(harbor/'visual-training-source',source)
    staged=harbor/'augmentation-source'
    for name in ['training.py','training_augmentation.py']:shutil.copyfile(staged/name,source/'aerorozum'/name)
    parent=root/'artifacts/training/submission-visual-projection-v2/updates-480/google_gemma-4-12B-it/stream_grpo/seed-17/adapter'
    assert (parent/'adapter_model.safetensors').is_file()
    # This is a dedicated benchmark base server. The production/refined server
    # on GPU 0 and ongoing experiments on GPUs 1/3 are unaffected.
    for service in ['aerorozum-gemma-native-base-v2-proxy','aerorozum-gemma-native-base-proxy',
                    'aerorozum-gemma-base-opencode-proxy','aerorozum-gemma-base-control']:
        subprocess.run(['systemctl','stop',service],check=False)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if uuids[2] in active:raise RuntimeError('Another process owns GPU 2; do not launch overlapping training')
    namespace='submission-geometric-augmentation-v1/updates-480';out=root/'artifacts'/namespace;out.mkdir(parents=True,exist_ok=True)
    spec=dict(namespace=namespace,manifest=str(root/'artifacts/submission-refinement-v1/train.json'),
        validation=str(root/'artifacts/pipeline/validation.json'),output=str(out),learning_rate=3e-5,
        sft_steps=480,grpo_steps=96,warm_start=str(parent),lora_scope='language_visual_projection',
        augmentation='training-hv-flips-v1',scope='Development perception only; unchanged source labels; test zones untouched')
    path=out/'specification.json'
    if path.exists():raise RuntimeError('Experiment already recorded; inspect before retrying')
    path.write_text(json.dumps(spec,indent=2)+'\n')
    args=['systemd-run','--unit=aerorozum-geometric-augmentation-v1-gpu2',
        '--property=RuntimeMaxSec=10800','--property=CPUQuota=300%','--property=MemoryMax=64G',
        f'--working-directory={source}',f'--property=StandardOutput=append:{out}/worker.log',
        f'--property=StandardError=append:{out}/worker.log']
    env=dict(CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(root/'artifacts'),
        AEROROZUM_LOCAL_MODEL_PATH=str(root/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
        AEROROZUM_LORA_SCOPE='language_visual_projection',AEROROZUM_TRAINING_AUGMENTATION='training-hv-flips-v1',
        HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    args.extend('--setenv='+k+'='+v for k,v in env.items())
    args.extend([str(root/'venv/bin/python'),'-u',str(root/'deployment/recovery/refine_perception.py'),str(path)])
    subprocess.run(args,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-job',default='opencode-gemma-native-base-v2');args=parser.parse_args();main(args.base_job)
