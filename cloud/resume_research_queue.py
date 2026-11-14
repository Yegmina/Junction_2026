"""Resume the original continuous queue after submission, preserving its protocol.

Only the dispatch pause and scheduling deadline change. Dataset, seed, reward,
physics and promotion settings remain intact. No running worker is interrupted.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys


def transition(settings,controls,state,deadline):
    if any(j.get('status')=='running' for key in ['jobs','evaluation_jobs'] for j in state.get(key,[])):
        raise ValueError('A managed worker is running; do not restart its coordinator')
    if controls.get('pause_dispatch') is not True:raise ValueError('Dispatch is already enabled; inspect its authoritative workers')
    parsed=datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:raise ValueError('An explicit UTC-offset deadline is required')
    revised_settings=copy.deepcopy(settings);revised_controls=copy.deepcopy(controls)
    revised_settings['deadline_utc']=parsed.astimezone(timezone.utc).isoformat()
    revised_controls.update(pause_dispatch=False,last_resume_reason='Submission artifacts frozen and published; resume previously authorized continuous research')
    revised_controls.pop('pause_reason',None)
    return revised_settings,revised_controls


def resume(root,deadline):
    root=Path(root);data=root/'artifacts';pipeline=data/'pipeline';source=root/'source'
    sys.path.insert(0,str(source))
    from aerorozum.common import save,utcnow
    from aerorozum.training import load_stream_manifest
    settings_path=root/'deployment/stream-settings.json';controls_path=pipeline/'controls.json';state_path=pipeline/'coordinator.json'
    settings=json.loads(settings_path.read_text());controls=json.loads(controls_path.read_text());state=json.loads(state_path.read_text())
    revised_settings,revised_controls=transition(settings,controls,state,deadline)
    if (datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds()<3600:raise ValueError('Less than one hour before the reviewed VM cutoff')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if any(uuids[g] in used for g in [1,2,3]):raise ValueError('A training GPU still has a live process')
    manifests={j['manifest'] for j in state['jobs'] if j['status']=='queued'}
    validation=[]
    for manifest in sorted(manifests):
        rows,protocol=load_stream_manifest(manifest,'stream_sft',root=data)
        validation.append(dict(manifest=manifest,images=len(rows),manifest_hash=protocol['manifest_hash']))
    folder=pipeline/'recovery'/('postsubmission-resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'settings-before.json',settings);save(folder/'controls-before.json',controls)
    save(folder/'queued-before.json',[j for j in state['jobs'] if j['status']=='queued'])
    service='aerorozum-stream-pipeline.service'
    subprocess.run(['systemctl','stop',service],check=True)
    current=json.loads(state_path.read_text());transition(settings,controls,current,deadline)
    save(settings_path,revised_settings)
    # The controller starts with dispatch still paused and the corrected cutoff.
    subprocess.run(['systemctl','start',service],check=True)
    if subprocess.run(['systemctl','is-active','--quiet',service]).returncode:
        raise RuntimeError('Coordinator did not start; leave dispatch paused and inspect logs')
    save(controls_path,revised_controls)
    receipt=dict(at=utcnow(),status='continuous_dispatch_resumed',queued_jobs=sum(j['status']=='queued' for j in current['jobs']),
        manifests=validation,settings_changes=['deadline_utc'],controls_changes=['pause_dispatch','pause_reason','last_resume_reason'],
        deadline_utc=revised_settings['deadline_utc'],submission_artifacts_changed=False,physics_gates_changed=False,
        training_recipe_changed=False,active_workers_interrupted=False)
    save(folder/'receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/opt/aerorozum');p.add_argument('--deadline-utc',required=True);a=p.parse_args();resume(a.root,a.deadline_utc)
