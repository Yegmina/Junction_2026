"""Separate canonical-tool/16k-context arm after the legacy Gemma cohort settles.

The protocol and context allowance change together. This is an infrastructure
and agent-interface diagnostic, not a causal model-training comparison.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts';NIGHT=ART/'overnight-20261115T000942'
PACKAGE=ROOT/'opencode-wizard-smoke/aerorozum-submission'
HARBOR=ROOT/'opencode-wizard-smoke/harbor-venv/bin/harbor'
END=datetime.fromisoformat('2026-11-15T05:45:00+00:00')


def active(unit):
    return subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True,timeout=15).stdout.strip() in ('active','activating')


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def wait(unit):
    while active(unit):time.sleep(20)


def main():
    from aerorozum.common import save,utcnow
    from cloud.overnight_missions import terminal_result,collect
    out=NIGHT/'native-chat-16k-v1';out.mkdir(parents=True,exist_ok=True)
    code=PACKAGE/'run_harbor_trial.py';proxy=ROOT/'deployment/recovery/gemma_openai_proxy_native.py'
    source=ROOT/'postsubmission/native-chat-control-v1';source.mkdir(parents=True,exist_ok=True)
    original=code.read_text();needle="key_name='OPENAI_API_KEY';context=24576;output=4096"
    if original.count(needle)!=1:raise ValueError('Unexpected frozen wrapper source')
    wrapper=source/'run_harbor_trial.py';changed=original.replace(needle,"key_name='OPENAI_API_KEY';context=16384;output=4096")
    if wrapper.exists() and wrapper.read_text()!=changed:raise ValueError('Preserve an already frozen control wrapper')
    wrapper.write_text(changed)
    protocol=dict(at=utcnow(),worlds=[dict(seed=seed,phase=phase) for seed in range(940201,940211) for phase in ['dawn','warming']],
        original_wrapper_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),control_wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        proxy_sha256=hashlib.sha256(proxy.read_bytes()).hexdigest(),context_tokens=16384,output_tokens=4096,flight_budget_s=75,
        model='deployed EF19 language adapter, unchanged',native_tool_protocol=True,original_submission_unchanged=True,
        baseline_outcomes_and_errors_preserved=True,score_based_retries=False,test_accessed=False,
        dispatch_deadline=END.isoformat(),interpretation='Canonical tool protocol and smaller context change together. This is not an isolated model or protocol effect. All declared outcomes, infrastructure failures and unstarted cases are retained.')
    target=out/'protocol.json'
    if target.exists():protocol=read(target)
    else:save(target,protocol)
    while datetime.now(timezone.utc)<END:
        settled=read(NIGHT/'native-gemma-controller-result.json').get('status')=='settled'
        busy=subprocess.check_output(['systemctl','list-units','--state=running','--no-legend','aerorozum-overnight-gemma-local-deployed-*'],text=True,timeout=15).strip()
        if settled and not busy:break
        save(out/'status.json',dict(at=utcnow(),status='waiting_for_legacy_cohort',legacy_controller_settled=settled))
        time.sleep(30)
    outcomes=[]
    for world in protocol['worlds']:
        if datetime.now(timezone.utc)>=END:break
        phase,seed=world['phase'],world['seed'];name=f'overnight-gemma-native16k-{phase}-{seed}'
        job=ROOT/'harbor/jobs'/name;proxy_unit='aerorozum-native16k-proxy-'+phase+'-'+str(seed);unit='aerorozum-'+name
        if job.exists():
            outcomes.append(collect('gemma-native16k',phase,seed));continue
        with urllib.request.urlopen('http://127.0.0.1:18940/health',timeout=15) as response:health=json.load(response)
        if health.get('ready') is not True or health.get('adapter_sha256')!='ef19fdcda7ed07979d3aca38a197af66dfe2d68702f3b40467dbfc9b9b68506a':raise ValueError('Expected deployed model is unavailable; no checkpoint substitution')
        logs=out/name;logs.mkdir(parents=True,exist_ok=True)
        command=['systemd-run','--unit='+proxy_unit,'--property=RuntimeMaxSec=2800','--property=CPUQuota=100%','--property=MemoryMax=2G',
            '--setenv=AEROROZUM_NATIVE_CHAT=1','--setenv=AEROROZUM_PROXY_HOST=172.17.0.1','--setenv=AEROROZUM_PROXY_PORT=18945',
            '--setenv=AEROROZUM_PROXY_MODEL=gemma-4-12b-refined-grpo-native16k','--setenv=AEROROZUM_PROXY_LOGS='+str(logs/'proxy'),'/usr/bin/python3',str(proxy)]
        subprocess.run(command,check=True,timeout=30)
        try:
            for attempt in range(20):
                try:
                    with urllib.request.urlopen('http://172.17.0.1:18945/v1/models',timeout=2) as response:
                        if not json.load(response).get('data'):raise ValueError('No proxy model')
                    break
                except Exception:time.sleep(.25)
            else:raise ValueError('Separate native-tool proxy failed to start')
            if not (out/'native-protocol-probe.json').exists():
                payload=dict(model='gemma-4-12b-refined-grpo-native16k',stream=False,max_tokens=64,
                    messages=[dict(role='user',content='Reply with READY only.')])
                request=urllib.request.Request('http://172.17.0.1:18945/v1/chat/completions',
                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=240) as response:probe=json.load(response)
                if not isinstance(probe.get('choices',[{}])[0].get('message'),dict):raise ValueError('Invalid native protocol probe')
                save(out/'native-protocol-probe.json',dict(at=utcnow(),request=payload,response=probe,
                    no_task_or_reference_data_used=True,no_drone_command_executed=True))
            save(out/'status.json',dict(at=utcnow(),status='running',seed=seed,phase=phase,completed=len(outcomes)))
            command=['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=2700','--property=CPUQuota=200%','--property=MemoryMax=8G',
                '--property=StandardOutput=append:'+str(logs/'worker.log'),'--property=StandardError=append:'+str(logs/'worker.log'),
                str(HARBOR.with_name('python')),str(wrapper),'--task',str(PACKAGE/'thermal-investigation-budgeted'),
                '--job-name',name,'--jobs-dir',str(ROOT/'harbor/jobs'),'--harbor',str(HARBOR),'--local-gemma',
                '--local-model','gemma-4-12b-refined-grpo-native16k','--base-url','http://host.docker.internal:18945/v1',
                '--seed',str(seed),'--phase',phase,'--budget','75','--output-tokens','4096']
            subprocess.run(command,check=True,timeout=30);wait(unit)
            report=collect('gemma-native16k',phase,seed);outcomes.append(report)
            save(logs/'control-result.json',report)
        finally:subprocess.run(['systemctl','stop',proxy_unit],timeout=30,check=False)
    save(out/'report.json',dict(at=utcnow(),status='settled',attempted=len(outcomes),planned=len(protocol['worlds']),
        unstarted=len(protocol['worlds'])-len(outcomes),test_accessed=False,interpretation=protocol['interpretation']))


if __name__=='__main__':main()
