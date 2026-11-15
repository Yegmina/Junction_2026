#!/usr/bin/env python3
"""Publish an individual paired field only after all executed direct gates pass."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from admission import certify_direct


def write_atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def assess(root,group,campaign_id,split):
    case=root/'cases'/group
    required=['present','time-strict','grid','domain','regularization','absent','absent-time','absent-grid','absent-domain']
    missing=[name for name in required if not (case/name/'field.npz').is_file() or not (case/name/'run-status.json').is_file() or json.loads((case/name/'run-status.json').read_text()).get('status')!='completed']
    marker=case/'pair-acceptance.json'
    if missing:
        result=dict(status='waiting_for_full_references',accepted=False,research_accepted=False,campaign_id=campaign_id,missing_completed_fields=missing)
        write_atomic(marker,result);return result
    reports={}
    for present in [True,False]:
        name='present' if present else 'absent';prefix='' if present else 'absent-'
        kwargs=dict(field=case/name/'field.npz',time_refined=case/('time-strict' if present else 'absent-time')/'field.npz',grid_refined=case/('grid' if present else 'absent-grid')/'field.npz',domain_refined=case/(prefix+'domain')/'field.npz',regularized=case/('regularization' if present else 'absent')/'field.npz',upstream_root=root/'upstream',output=case/name/'acceptance-report.json',publish=True)
        reports[name]=certify_direct(**kwargs)
    p=case/'present/field.npz';a=case/'absent/field.npz'
    with np.load(p) as present,np.load(a) as absent:
        matching_axes=all(np.array_equal(present[key],absent[key]) for key in ['x_m','y_m'])
        coverage=[float(present['times_s'][0]),float(present['times_s'][-1])]
        matching_coverage=coverage==[float(absent['times_s'][0]),float(absent['times_s'][-1])]
        bounds=dict(x_m=[float(present['x_m'][0]),float(present['x_m'][-1])],y_m=[float(present['y_m'][0]),float(present['y_m'][-1])])
    pm=json.loads(p.with_suffix('.json').read_text());am=json.loads(a.with_suffix('.json').read_text())
    same_sources=pm.get('application_source_hashes')==am.get('application_source_hashes')
    accepted=all(r['accepted'] for r in reports.values()) and matching_axes and matching_coverage and same_sources
    result=dict(status='numerically_accepted_pair' if accepted else 'numerical_pair_admission_failed',accepted=accepted,research_accepted=accepted,campaign_id=campaign_id,present_path=str(p),absent_path=str(a),present_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),absent_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),present_acceptance_report=str(case/'present/acceptance-report.json'),absent_acceptance_report=str(case/'absent/acceptance-report.json'),matching_spatial_axes=matching_axes,matching_physical_time_coverage=matching_coverage,physical_time_coverage_s=coverage,native_time_grids='Independent adaptive solver states; interpolate both at the same physical capture time without extrapolation',source_hashes_match=same_sources,cell_centre_bounds=bounds,field_validation='not_performed',surrogate_admission='separate_error_gate_not_implied',direct_reports={name:dict(accepted=r['accepted'],failed_checks=[c['name'] for c in r['checks'] if c['status'] not in ('passed','not_applicable')]) for name,r in reports.items()})
    write_atomic(marker,result)
    if accepted:
        record=dict(campaign_id=campaign_id,split=split,field_path=str(p),counterfactual_path=str(a),acceptance_report_path=str(marker),status='accepted',present_sha256=result['present_sha256'],absent_sha256=result['absent_sha256'])
        with (root/'registry.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);registry=root/'registry.json'
            data=json.loads(registry.read_text()) if registry.exists() else dict(fields=[])
            data['fields']=[x for x in data['fields'] if x.get('campaign_id')!=campaign_id]+[record]
            write_atomic(registry,data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'));p.add_argument('--group',default='qualified-skin2-full');p.add_argument('--campaign-id',default='dumux-reference-20261114-000');p.add_argument('--split',default='train');p.add_argument('--watch',type=float,default=30)
    a=p.parse_args()
    if not (a.root/'registry.json').exists():write_atomic(a.root/'registry.json',dict(fields=[]))
    while True:
        try:result=assess(a.root,a.group,a.campaign_id,a.split);print(json.dumps(result),flush=True)
        except Exception as error:result=dict(status='admission_error',accepted=False,error=f'{type(error).__name__}: {error}');print(json.dumps(result),flush=True)
        if result['status'] in ('numerically_accepted_pair','numerical_pair_admission_failed'):return 0 if result['accepted'] else 1
        if not a.watch:return 0
        time.sleep(a.watch)


if __name__=='__main__':raise SystemExit(main())
