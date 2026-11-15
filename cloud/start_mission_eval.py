"""Start four bounded isolated inference workers on the existing VM only."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['paired','blank'],default='paired');a=p.parse_args();condition=a.condition
    root=Path('/opt/aerorozum/mission-decision-eval-v1');corpus=root/('input' if condition=='paired' else 'input-blank')/'corpus.json';out=root/('results' if condition=='paired' else 'results-blank');out.mkdir(parents=True,exist_ok=True)
    training=Path('/opt/aerorozum/artifacts/training');model='google_gemma-4-12B-it'
    real=training/'real-transfer-machine-labels-v1'/model
    stream=training/'incremental-stream-v1/e2c068fbee9afe0546bf4fa9'/model
    variants=[('base',None),('real_sft',real/'real_sft/seed-17/adapter'),('real_grpo',real/'real_grpo/seed-17/adapter'),('stream_grpo',stream/'stream_grpo/seed-17/adapter')]
    for name,adapter in variants:
        if adapter:
            summary=json.loads((adapter.parent/'summary.json').read_text())
            if summary['status']!='completed' or summary['seed']!=17:raise ValueError('Unverified adapter lineage')
            if not (adapter/'adapter_model.safetensors').is_file():raise ValueError('Adapter bytes missing')
    for gpu,(name,adapter) in enumerate(variants):
        destination='gs://aerorozum-thermal-rl-avid-relic-509408-e4/mission-eval/'+('paired-decisions-v1' if condition=='paired' else 'paired-decisions-blank-v1')+'/'+name
        folder=out/name;folder.mkdir(exist_ok=True)
        unit='aerorozum-decision-eval-'+name.replace('_','-')+('-v1' if condition=='paired' else '-blank-v1')
        args=['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=1200','--property=CPUQuota=200%','--property=MemoryMax=64G','--property=Restart=no','--property=StandardOutput=append:'+str(folder/'worker.log'),'--property=StandardError=append:'+str(folder/'worker.log'),'--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2','--setenv=TOKENIZERS_PARALLELISM=false','--setenv=HF_HUB_OFFLINE=1','--setenv=HF_HOME=/opt/aerorozum/hf','/opt/aerorozum/venv/bin/python','-u',str(root/'mission_decision_eval.py'),'--corpus',str(corpus),'--output',str(folder),'--variant',name,'--batch-size','4','--gcs',destination]
        if adapter:args+=['--adapter',str(adapter)]
        subprocess.run(args,check=True);print(json.dumps(dict(unit=unit,gpu=gpu,variant=name,max_seconds=1200)),flush=True)

if __name__=='__main__':main()
