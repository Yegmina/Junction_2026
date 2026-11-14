"""Test native tool decisions without a thinking channel, on the same adapter."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import uuid


def main():
    root=Path('/opt/aerorozum');h=root/'harbor'
    for job in ['opencode-gemma-native-refined-v2','opencode-gemma-native-base-v2']:
        result=json.loads((h/'jobs'/job/'result.json').read_text())
        if not result.get('finished_at'):raise RuntimeError('Previous native comparison is still active; do not restart its model')
    for _ in range(60):
        connections=subprocess.check_output(['ss','-Hnt','state','established','(','sport','=',':18940',')'],text=True).strip()
        utilization=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
        if not connections and utilization<5:break
        time.sleep(5)
    else:raise RuntimeError('Inference is busy; leave the active request intact')
    folder=h/'fast-policy-control';folder.mkdir(exist_ok=True)
    config=Path('/etc/systemd/system/aerorozum-cloud-gemma.service.d/native-fast-profile.conf')
    if config.exists():raise RuntimeError('Fast profile already recorded; inspect the existing trial before retrying')
    config.write_text('[Service]\nEnvironment=AEROROZUM_NATIVE_THINKING=0\n')
    subprocess.run(['systemctl','daemon-reload'],check=True)
    subprocess.run(['systemctl','restart','aerorozum-cloud-gemma'],check=True)
    for _ in range(150):
        try:
            health=json.load(urllib.request.urlopen('http://127.0.0.1:18940/health',timeout=3))
            if health.get('adapter_sha256')=='ef19fdcda7ed07979d3aca38a197af66dfe2d68702f3b40467dbfc9b9b68506a':break
        except Exception:pass
        time.sleep(2)
    else:
        config.unlink();subprocess.run(['systemctl','daemon-reload'],check=True)
        subprocess.run(['systemctl','restart','aerorozum-cloud-gemma'],check=True)
        raise RuntimeError('Fast-profile checkpoint gate failed; restored previous configuration')
    payload=dict(messages=[dict(role='user',content='Use bash to run drone status.')],tools=[dict(type='function',function=dict(name='bash',description='Execute command',parameters=dict(type='object',properties=dict(command=dict(type='string')),required=['command'])))],max_tokens=512)
    req=urllib.request.Request('http://127.0.0.1:18940/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Aero-Native-Session':uuid.uuid4().hex*2})
    with urllib.request.urlopen(req,timeout=900) as response:probe=json.load(response)
    assert probe['aerorozum']['thinking_enabled'] is False
    (folder/'probe.json').write_text(json.dumps(probe,indent=2))
    (folder/'health.json').write_text(json.dumps(health,indent=2))
    source=folder/'proxy.py';shutil.copyfile('/tmp/gemma_openai_proxy.py',source)
    proxy='aerorozum-gemma-fast-policy-proxy'
    subprocess.run(['systemd-run','--unit='+proxy,'--property=RuntimeMaxSec=5000','--property=MemoryMax=1G',
        '--setenv=AEROROZUM_PROXY_HOST=172.17.0.1','--setenv=AEROROZUM_PROXY_PORT=18957',
        '--setenv=AEROROZUM_PROXY_MODEL=gemma-4-12b-refined-no-thinking','--setenv=AEROROZUM_NATIVE_CHAT=1',
        '--setenv=AEROROZUM_INFERENCE_TIMEOUT_S=900','--setenv=AEROROZUM_INFERENCE_URL=http://127.0.0.1:18940/v1/chat/completions',
        '--setenv=AEROROZUM_PROXY_LOGS='+str(folder/'proxy'),'/usr/bin/python3',str(source)],check=True)
    job='opencode-gemma-fast-v8-dawn180'
    subprocess.run(['systemd-run','--unit=aerorozum-harbor-'+job,'--property=RuntimeMaxSec=2600','--property=MemoryMax=20G',
        '/usr/bin/python3','/tmp/run_harbor_trial_configurable.py','--task',str(h/'source-v8/harbor/thermal-investigation'),
        '--jobs-dir',str(h/'jobs'),'--harbor',str(h/'venv/bin/harbor'),'--job-name',job,'--seed','414331','--phase','dawn','--budget','180',
        '--local-gemma','--local-model','gemma-4-12b-refined-no-thinking','--base-url','http://host.docker.internal:18957/v1'],check=True)
    (folder/'protocol.json').write_text(json.dumps(dict(adapter_sha256=health['adapter_sha256'],thinking_enabled=False,request_timeout_s=900,
        task='source-v8',phase='dawn',budget_s=180,seed=414331,scope='Joint native inference-profile ablation; not causal post-training improvement'),indent=2))


if __name__=='__main__':main()
