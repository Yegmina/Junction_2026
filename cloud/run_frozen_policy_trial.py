"""Run an explicitly frozen adapter through native OpenCode on a free GPU.

The receipt binds checkpoint, task path and inference settings before launch.
This changes inference configuration, never task rules or an active checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def active(unit):return subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0


def run(args):
    root=Path('/opt/aerorozum');h=root/'harbor';data=root/'artifacts'
    parent=Path(args.parent_result).resolve();record=json.loads(parent.read_text())
    if not parent.is_relative_to(data) or record.get('status')!='completed':
        raise ValueError('Need a completed parent result inside the artifact store')
    adapter=(data/record['results']['grpo']['adapter']).resolve()
    if not adapter.is_relative_to(data/'training'):raise ValueError('Adapter outside training store')
    weight=adapter/'adapter_model.safetensors'
    if sha(weight)!=args.adapter_sha256:raise ValueError('Frozen checkpoint hash differs')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if uuids[args.gpu] in used:raise RuntimeError('Requested GPU is occupied; leave its process alone')
    folder=h/args.job
    if folder.exists() or (h/'jobs'/args.job).exists():raise RuntimeError('Trial identity already exists')
    for host,port in [('127.0.0.1',args.port),('172.17.0.1',args.proxy_port)]:
        with socket.socket() as probe:probe.bind((host,port))
    folder.mkdir();source=h/'native-service-source-v2/cloud';service='aerorozum-policy-'+args.job
    proxy=service+'-proxy';unit='aerorozum-harbor-'+args.job
    receipt=dict(parent_result=str(parent),parent_result_sha256=sha(parent),adapter=str(adapter),
        adapter_sha256=sha(weight),task=str(Path(args.task).resolve()),seed=args.seed,
        phase=args.phase,budget=args.budget,gpu=args.gpu,thinking=False,proxy_timeout_s=900,
        max_output_tokens=4096,context_tokens=24576,scope='Frozen native-policy development trial; no test selection or superiority claim')
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    env=dict(CUDA_VISIBLE_DEVICES=str(args.gpu),AEROROZUM_GEMMA_WEIGHTS=str(root/'models/gemma-4-12B-it'),
        AEROROZUM_GEMMA_ADAPTER=str(adapter),AEROROZUM_MISSION_MODEL='gemma-4-12b-visual-frozen',
        AEROROZUM_NATIVE_THINKING='0',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    command=['systemd-run','--unit='+service,'--property=RuntimeMaxSec=3400','--property=CPUQuota=300%',
        '--property=MemoryMax=60G','--working-directory='+str(source)]
    command+=['--setenv='+k+'='+v for k,v in env.items()]
    command+=[str(root/'venv/bin/python'),'-m','uvicorn','gemma_service:app','--host','127.0.0.1','--port',str(args.port)]
    subprocess.run(command,check=True)
    try:
        for _ in range(150):
            try:
                health=json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=3))
                if health.get('adapter_sha256')==args.adapter_sha256 and health.get('native_tool_protocol')=='gemma4-native-tools-v2':break
            except Exception:pass
            time.sleep(2)
        else:raise RuntimeError('Checkpoint and native-protocol health gate failed')
        (folder/'health.json').write_text(json.dumps(health,indent=2))
        subprocess.run(['systemd-run','--unit='+proxy,'--property=RuntimeMaxSec=3100','--property=MemoryMax=1G',
            '--setenv=AEROROZUM_PROXY_HOST=172.17.0.1','--setenv=AEROROZUM_PROXY_PORT='+str(args.proxy_port),
            '--setenv=AEROROZUM_PROXY_MODEL=gemma-4-12b-visual-frozen','--setenv=AEROROZUM_NATIVE_CHAT=1',
            '--setenv=AEROROZUM_INFERENCE_TIMEOUT_S=900',
            '--setenv=AEROROZUM_INFERENCE_URL='+f'http://127.0.0.1:{args.port}/v1/chat/completions',
            '--setenv=AEROROZUM_PROXY_LOGS='+str(folder/'proxy'),'/usr/bin/python3',str(Path(__file__).with_name('gemma_openai_proxy.py'))],check=True)
        subprocess.run(['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=2600','--property=MemoryMax=20G',
            '/usr/bin/python3',str(Path(__file__).with_name('run_harbor_trial.py')),'--task',args.task,
            '--jobs-dir',str(h/'jobs'),'--harbor',str(h/'venv/bin/harbor'),'--job-name',args.job,
            '--seed',str(args.seed),'--phase',args.phase,'--budget',str(args.budget),'--local-gemma',
            '--local-model','gemma-4-12b-visual-frozen','--base-url',f'http://host.docker.internal:{args.proxy_port}/v1'],check=True)
        while active(unit):time.sleep(10)
    finally:
        if active(unit):raise RuntimeError('Trial still active; leave its inference services running')
        subprocess.run(['systemctl','stop',proxy,service],check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-result',required=True);p.add_argument('--adapter-sha256',required=True)
    p.add_argument('--task',required=True);p.add_argument('--job',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--phase',default='dawn');p.add_argument('--budget',type=int,default=75)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--port',type=int,default=18946);p.add_argument('--proxy-port',type=int,default=18958)
    run(p.parse_args())
