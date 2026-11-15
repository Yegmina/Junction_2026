"""Resume a declared native Harbor comparison; never alter tasks or hide failures."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/opt/aerorozum')
STUDY=ROOT/'artifacts/overnight-20261115T000942'
PACKAGE=ROOT/'opencode-wizard-smoke/aerorozum-submission'
HARBOR=ROOT/'opencode-wizard-smoke/harbor-venv/bin/harbor'
PYTHON=HARBOR.with_name('python')
JOBS=ROOT/'harbor/jobs'
DEADLINE=datetime.fromisoformat('2026-11-15T05:45:00+00:00')


def terminal_result(path):
    try:
        result=json.loads(Path(path).read_text())
    except (OSError,ValueError):
        return None
    stats=result.get('stats',{})
    if not result.get('finished_at') or stats.get('n_running_trials') or stats.get('n_pending_trials'):
        return None
    return result


def save(path,value):
    from aerorozum.common import save as atomic_save
    atomic_save(path,value)


def collect(model,phase,seed):
    name=f'overnight-{model}-{phase}-{seed}'
    job=JOBS/name
    report=dict(model=model,phase=phase,seed=seed,job=str(job),
                status='terminal_without_result',test_accessed=False)
    result=terminal_result(job/'result.json')
    if result is not None:
        report.update(status='errored' if result.get('stats',{}).get('n_errored_trials') else 'completed',harbor_result=result)
    trials=[]
    for path in job.glob('*/result.json'):
        row=json.loads(path.read_text())
        ledger=path.parent/'verifier/ledger.json'
        check=json.loads(ledger.read_text()) if ledger.exists() else None
        trials.append(dict(path=str(path),result=row,ledger=check))
    report['trials']=trials
    save(STUDY/name/'research-result.json',report)
    return report


def play(model,phase,seed):
    name=f'overnight-{model}-{phase}-{seed}';unit='aerorozum-'+name
    folder=STUDY/name;folder.mkdir(parents=True,exist_ok=True)
    result=JOBS/name/'result.json'
    if terminal_result(result) is not None:
        return collect(model,phase,seed)
    active=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip()
    if active not in ('active','activating'):
        # An existing partial directory is retained as a failed attempt. A new
        # seed is more informative than silently replaying an unfavorable run.
        if (JOBS/name).exists():
            return collect(model,phase,seed)
        command=['systemd-run','--unit='+unit,'--property=RuntimeMaxSec=2700',
            '--property=CPUQuota=200%','--property=MemoryMax=8G',
            '--property=StandardOutput=append:'+str(folder/'worker.log'),
            '--property=StandardError=append:'+str(folder/'worker.log')]
        if model=='oracle':
            command += ['--setenv=AEROROZUM_TASK_SEED='+str(seed),
                '--setenv=AEROROZUM_TASK_PHASE='+phase,'--setenv=AEROROZUM_TASK_BUDGET=75',
                str(HARBOR),'run','-p',str(PACKAGE/'thermal-investigation-budgeted'),
                '-a','oracle','-o',str(JOBS),'--job-name',name,'-n','1']
        else:
            command += [str(PYTHON),str(PACKAGE/'run_harbor_trial.py'),
                '--task',str(PACKAGE/'thermal-investigation-budgeted'),'--job-name',name,
                '--jobs-dir',str(JOBS),'--harbor',str(HARBOR),'--model',model,
                '--key-file','/etc/aerorozum/hackathon-key','--seed',str(seed),
                '--phase',phase,'--budget','75','--output-tokens','4096' if model=='gemma-local-deployed' else '32768']
            if model=='gemma-local-deployed':command+=['--local-gemma']
        subprocess.run(command,check=True,timeout=30)
    while True:
        state=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True,timeout=15).stdout.strip()
        if state not in ('active','activating'):
            break
        time.sleep(20)
    return collect(model,phase,seed)


def main(local_gemma=False):
    STUDY.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
        worlds=[dict(seed=seed,phase=phase) for seed in range(940201,940211) for phase in ['dawn','warming']],
        models=['gemma-local-deployed'] if local_gemma else ['gemini-4-carbon','gemini-4-argon','oracle'],budget=75,output_tokens=4096 if local_gemma else 32768,
        dispatch_deadline=DEADLINE.isoformat(),selection='Report every declared outcome; no outcome-based retries',
        no_model_training_claimed=True,original_submission_unchanged=True)
    protocol['model_context_tokens']=24576 if local_gemma else 131072
    protocol['controller_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if local_gemma:protocol['model_note']='Currently deployed language adapter; not the selected visual-perception checkpoint. Inference budgets differ from frontier arms.'
    path=STUDY/('native-gemma-protocol.json' if local_gemma else 'native-missions-protocol.json')
    if not path.exists():save(path,protocol)
    else:protocol=json.loads(path.read_text())
    def worker(model):
        for world in protocol['worlds']:
            if datetime.now(timezone.utc)>=DEADLINE:break
            try:
                report=play(model,world['phase'],world['seed'])
                # Metadata API authenticates archival only. Credentials and
                # OpenCode application-cache files are never uploaded.
                from cloud.airborne_worker import metadata_record_store
                store=metadata_record_store('aerorozum-thermal-rl-avid-relic-509408-e4',prefix='research/overnight-20261115/native-missions')
                key=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
                uri=store.put(key,report,'progress')
                save(STUDY/f'latest-{model}.json',dict(at=datetime.now(timezone.utc).isoformat(),uri=uri,
                    seed=world['seed'],phase=world['phase'],status=report['status']))
            except Exception as error:
                save(STUDY/f'controller-error-{model}-{world["seed"]}-{world["phase"]}.json',
                     dict(error=str(error)[:1500],at=datetime.now(timezone.utc).isoformat()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker,protocol['models']))
    save(STUDY/('native-gemma-controller-result.json' if local_gemma else 'native-controller-result.json'),dict(status='settled',at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--local-gemma',action='store_true')
    main(parser.parse_args().local_gemma)
