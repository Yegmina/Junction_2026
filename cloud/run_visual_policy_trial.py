"""Use a released training GPU for a frozen, development-selected policy trial."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def active(unit):
    return subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0


def main():
    root=Path('/opt/aerorozum');h=root/'harbor';data=root/'artifacts'
    while active('aerorozum-visual-continuation-v1-gpu1'):time.sleep(10)
    completed=data/'submission-visual-continuation-v1/updates-240/result.json'
    if not completed.is_file() or json.loads(completed.read_text()).get('status')!='completed':
        raise RuntimeError('Training did not finish cleanly; inspect before scheduling its GPU')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if uuids[1] in used:raise RuntimeError('GPU 1 still has a compute process; do not interrupt it')
    choices=[]
    for family in ['submission-visual-projection-v2','submission-visual-continuation-v1']:
        for file in (data/family).rglob('result.json'):
            record=json.loads(file.read_text())
            if record.get('status')!='completed':continue
            result=record['results'];score=result['grpo_evaluation'];adapter=data/result['grpo']['adapter']
            choices.append(dict(source_result=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                adapter=str(adapter),adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
                micro_f2=score['micro_f2'],negative_false_alarm_rate=score['negative_false_alarm_rate'],
                valid_json_fraction=score['valid_json_fraction']))
    selected=max(choices,key=lambda c:(c['micro_f2'],-c['negative_false_alarm_rate'],c['valid_json_fraction']))
    folder=h/'visual-policy-control';folder.mkdir(exist_ok=True)
    receipt=folder/'selection.json'
    if receipt.exists():raise RuntimeError('Policy selection already frozen; inspect its authoritative services before retrying')
    receipt.write_text(json.dumps(dict(scope='Development selection only; no sealed-test access',
        selection='Maximum GRPO development F2, then lower negative false-alarm rate, then valid JSON',
        candidates=choices,selected=selected),indent=2)+'\n')
    for host,port in [('127.0.0.1',18944),('172.17.0.1',18955)]:
        with socket.socket() as probe:probe.bind((host,port))
    source=h/'native-service-source-v2/cloud';service='aerorozum-gemma-visual-policy'
    args=['systemd-run','--unit='+service,'--property=RuntimeMaxSec=5400','--property=CPUQuota=300%',
        '--property=MemoryMax=60G',f'--working-directory={source}']
    env=dict(CUDA_VISIBLE_DEVICES='1',AEROROZUM_GEMMA_WEIGHTS=str(root/'models/gemma-4-12B-it'),
        AEROROZUM_GEMMA_ADAPTER=selected['adapter'],AEROROZUM_MISSION_MODEL='google/gemma-4-12B-it/visual-development-selected',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    args.extend('--setenv='+k+'='+v for k,v in env.items())
    args.extend([str(root/'venv/bin/python'),'-m','uvicorn','gemma_service:app','--host','127.0.0.1','--port','18944'])
    subprocess.run(args,check=True)
    proxy='aerorozum-gemma-visual-native-proxy';job='opencode-gemma-visual-v8-dawn180';unit='aerorozum-harbor-'+job
    try:
        for _ in range(150):
            try:
                health=json.load(urllib.request.urlopen('http://127.0.0.1:18944/health',timeout=3))
                if health.get('adapter_sha256')==selected['adapter_sha256'] and health.get('native_tool_protocol')=='gemma4-native-tools-v2':break
            except Exception:pass
            time.sleep(2)
        else:raise RuntimeError('Visual policy inference failed its checkpoint/protocol health gate')
        (folder/'health.json').write_text(json.dumps(health,indent=2))
        subprocess.run(['systemd-run','--unit='+proxy,'--property=RuntimeMaxSec=5000','--property=MemoryMax=1G',
            '--setenv=AEROROZUM_PROXY_HOST=172.17.0.1','--setenv=AEROROZUM_PROXY_PORT=18955',
            '--setenv=AEROROZUM_PROXY_MODEL=gemma-4-12b-visual-selected-native','--setenv=AEROROZUM_NATIVE_CHAT=1',
            '--setenv=AEROROZUM_INFERENCE_URL=http://127.0.0.1:18944/v1/chat/completions',
            '--setenv=AEROROZUM_PROXY_LOGS='+str(folder/'proxy'),'/usr/bin/python3',str(source/'gemma_openai_proxy.py')],check=True)
        subprocess.run(['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=2600','--property=MemoryMax=20G',
            '/usr/bin/python3','/tmp/run_harbor_trial_configurable.py','--task',str(h/'source-v8/harbor/thermal-investigation'),
            '--jobs-dir',str(h/'jobs'),'--harbor',str(h/'venv/bin/harbor'),'--job-name',job,'--seed','414331',
            '--phase','dawn','--budget','180','--local-gemma','--local-model','gemma-4-12b-visual-selected-native',
            '--base-url','http://host.docker.internal:18955/v1'],check=True)
        while active(unit):time.sleep(10)
    finally:
        if active(unit):raise RuntimeError('Policy trial is still active; leave its inference services running')
        subprocess.run(['systemctl','stop',proxy,service],check=False)


if __name__=='__main__':main()
