#!/usr/bin/env python3
"""Bounded existing-VM qualification controller; never changes error limits.

Keeps cloud jobs independent of the laptop. At most two further mesh/domain
attempts are permitted. An exact scene is published only by the full verifier.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def load(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def active(unit):return subprocess.run(['systemctl','is-active','--quiet',unit+'.service']).returncode==0


def published(root,group):
    """A passing report alone is not a completed publication transaction."""
    report=load(root/'cases'/group/'pair-acceptance.json',{})
    if not report.get('accepted') or report.get('status')!='numerically_accepted_pair':return False
    records=load(root/'registry.json',{}).get('fields',[])
    record=next((item for item in records if item.get('campaign_id')==report.get('campaign_id') and item.get('status')=='accepted'),None)
    if record is None:return False
    for name,key in [('present','field_path'),('absent','counterfactual_path')]:
        path=Path(report.get(name+'_path',''))
        expected=report.get(name+'_sha256')
        if not expected or record.get(name+'_sha256')!=expected or record.get(key)!=str(path) or not path.is_file():return False
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        metadata=load(path.with_suffix('.json'),{})
        if digest!=expected or metadata.get('field_sha256')!=expected or metadata.get('validation_state')!='numerically_accepted':return False
    return True


def launch(unit,command,log,cores=4,memory='8G',duration=18000):
    if active(unit):return
    subprocess.run(['systemctl','reset-failed',unit+'.service'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['systemd-run','--unit='+unit,'--setenv=PYTHONPATH=/opt/aerorozum/physics/source','--property=CPUQuota='+str(cores*100)+'%','--property=MemoryMax='+memory,'--property=RuntimeMaxSec='+str(duration),'/bin/bash','-lc',shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'],check=True)


def ensure(root,state):
    version=state['version'];group='axisymmetric-qualification-v'+str(version);folder=root/'cases'/group;folder.mkdir(parents=True,exist_ok=True)
    source=root/'source/physics';python='/opt/aerorozum/venv/bin/python'
    scale=state['mesh_scale'];radius=state['radius_m'];depth=state['depth_m']
    cases=[('present',[]),('absent',['--no-target']),('time',['--dt','15','--initial-dt','.5']),('absent-time',['--no-target','--dt','15','--initial-dt','.5']),('grid',['--refinement',str(2*scale)]),('absent-grid',['--no-target','--refinement',str(2*scale)]),('domain',['--radius',str(2*radius),'--depth',str(2*depth)]),('absent-domain',['--no-target','--radius',str(2*radius),'--depth',str(2*depth)]),('regularization',['--set','Target.Porosity=1e-7','--set','Target.Permeability=1e-25']),('correspondence-depth04',['--radius','1','--depth','.4']),('correspondence-absent-depth04',['--radius','1','--depth','.4','--no-target'])]
    for name,extra in cases:
        unit='aerorozum-physics-axisym-v'+str(version)+'-'+name;case=folder/name;prior=load(case/'run-status.json',{})
        if active(unit) or prior.get('status')=='completed':continue
        if prior and prior.get('native_status')=='completed':continue
        if prior:
            attempts=state.setdefault('restarted_cases',{}).get(unit,0)
            if attempts>=1:state['blocker']='A native solver failed again after one preserved restart: '+unit;return
            archived=case.with_name(case.name+'.interrupted-'+str(int(time.time())));case.rename(archived)
            prior.update(status='interrupted_preserved',research_accepted=False,interrupted_at=time.time());save(archived/'run-status.json',prior)
            save(archived/'retirement.json',dict(status='interrupted_preserved',research_accepted=False,reason='Cloud solver unit stopped before physical completion; restarting complete day in a fresh directory',recorded_at=time.time()))
            state['restarted_cases'][unit]=attempts+1
        command=[python,str(source/'run_axisymmetric.py'),'--root',str(root/'upstream'),'--output-dir',str(case),'--duration','86400','--threads','4','--refinement',str(scale),'--radius',str(radius),'--depth',str(depth),'--cartesian-cells','100',*extra]
        launch(unit,command,folder/(name+'-controller.log'))
    admission='aerorozum-physics-axisym-v'+str(version)+'-admission'
    report=load(folder/'admission-progress.json',{})
    if not active(admission) and not published(root,group):
        launch(admission,[python,str(source/'axisymmetric_admission.py'),'--root',str(root),'--group',group,'--watch','30','--publish'],root/('axisymmetric-v'+str(version)+'-admission.log'),2,'12G',60000)
    uploader='aerorozum-physics-axisym-v'+str(version)+'-upload'
    if not active(uploader):launch(uploader,[python,'-u',str(source/'upload_artifacts.py'),'--root',str(folder),'--prefix','physics/'+group,'--watch','60'],root/('axisymmetric-v'+str(version)+'-upload.log'),1,'2G',60000)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'));p.add_argument('--deadline',default='2026-11-15T05:15:00+00:00');p.add_argument('--max-version',type=int,default=4);a=p.parse_args();root=a.root
    status=root/'controller-state.json';state=load(status,dict(version=2,mesh_scale=1,radius_m=1.,depth_m=.8,history=[],status='running'))
    deadline=datetime.fromisoformat(a.deadline).timestamp()
    while time.time()<deadline:
        try:
            ensure(root,state)
            group='axisymmetric-qualification-v'+str(state['version']);report=load(root/'cases'/group/'admission-progress.json',{})
            complete=published(root,group)
            state.update(updated_at=time.time(),active_group=group,admission_status=report.get('status','starting'),accepted_pairs=1 if complete else 0)
            if complete:state['status']='accepted_exact_scene';save(status,state);return
            if report.get('status')=='admission_error':
                state['consecutive_admission_errors']=state.get('consecutive_admission_errors',0)+1
                if state['consecutive_admission_errors']>=3:
                    state.update(status='needs_review',blocker='The strict admission verifier repeatedly failed; no acceptance inferred',verifier_error=report.get('error'));save(status,state);return
            else:state['consecutive_admission_errors']=0
            if state.get('blocker'):state['status']='needs_review';save(status,state);return
            if report.get('status')=='axisymmetric_native_admission_failed':
                failed=report.get('failed_checks',[]);native=[x for x in failed if not x.startswith('actual_3d_')]
                spatial=any(x in ('temperature_absolute_present_grid','temperature_absolute_absent_grid') for x in native)
                domain=any(x in ('temperature_absolute_present_domain','temperature_absolute_absent_domain') for x in native)
                supported=all(('grid' in x or 'domain' in x) and not x.startswith('actual_extruded') for x in native)
                if state['version']<a.max_version and supported and (spatial or domain):
                    # Never race the last projections or kill an active native
                    # trajectory; preserve the entire completed failed cohort.
                    if any(active('aerorozum-physics-axisym-v'+str(state['version'])+'-'+name) for name in ['present','absent','grid','absent-grid','time','absent-time','domain','absent-domain','regularization']):
                        save(status,state);time.sleep(30);continue
                    previous=root/'cases'/group;save(previous/'superseded.json',dict(status='failed_superseded',research_accepted=False,failed_checks=failed,reason='Bounded stricter grid/domain resolution attempt; no tolerance changes',recorded_at=time.time()))
                    subprocess.run(['systemctl','stop','aerorozum-physics-axisym-v'+str(state['version'])+'-admission.service'])
                    state['history'].append(dict(group=group,failed_checks=failed,retired_at=time.time()))
                    state['version']+=1
                    if spatial:state['mesh_scale']*=2
                    if domain:state['radius_m']*=2;state['depth_m']*=2
                else:state.update(status='needs_review',blocker='Actual numerical gates failed outside the bounded automatic mesh/domain policy',failed_checks=failed);save(status,state);return
            elif report.get('status')=='axisymmetric_3d_correspondence_failed':
                state.update(status='needs_review',blocker='Native qualification passed but actual3D correspondence failed; no automatic gate removal',failed_checks=report.get('failed_checks'));save(status,state);return
            save(status,state)
        except Exception as error:state.update(last_error=f'{type(error).__name__}: {error}',updated_at=time.time());save(status,state)
        time.sleep(30)
    state.update(status='deadline_guard',accepted_pairs=0,blocker='Bounded deadline reached; preserve all data, no acceptance inferred');save(status,state)


if __name__=='__main__':main()
