#!/usr/bin/env python3
"""Compare every native float64 diagnostic against an executed CPU trajectory.

This is a numerical parallel-backend check, never a field-validation claim.
The compared time interval must be inside both actual trajectories. No states
are extrapolated and the native mesh/physical inputs must match.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from aerorozum.physics import assess_dumux_ledger


def read(case):
    state=json.loads((case/'run-status.json').read_text())
    description=json.loads((case/'surface-format.json').read_text())
    n=int(description['points']);columns=1+2*n
    path=case/'surface-frames.bin';count=path.stat().st_size//(columns*8)*columns
    values=np.fromfile(path,dtype='<f8' if description['byte_order']=='little' else '>f8',count=count).reshape(-1,columns)
    xy=np.loadtxt(case/'surface-grid.csv',delimiter=',',skiprows=1).reshape(-1,2)
    if not np.all(np.isfinite(values)) or len(values)<2 or np.any(np.diff(values[:,0])<=0):raise ValueError('Invalid native states')
    return state,xy,values


def verify(reference,candidate,duration_s=600.,evidence_dir=None):
    old,xy,a=read(reference);new,uv,b=read(candidate)
    same_input=old.get('input_sha256')==new.get('input_sha256')
    # A prefix benchmark differs only in its requested stopping time.
    p={k:v for k,v in old['parameters'].items() if k!='TimeLoop.TEnd'}
    q={k:v for k,v in new['parameters'].items() if k!='TimeLoop.TEnd'}
    same_parameters=p==q
    if xy.shape!=uv.shape or not np.allclose(xy,uv,rtol=0,atol=1e-12):raise ValueError('Native physical surface meshes differ')
    common_end=min(float(a[-1,0]),float(b[-1,0]))
    covers=bool(a[0,0]==0 and b[0,0]==0 and common_end>=duration_s)
    times=np.unique(np.r_[a[a[:,0]<=min(common_end,duration_s),0],b[b[:,0]<=min(common_end,duration_s),0],*([duration_s] if covers else [])])
    errors=np.abs(interp1d(a[:,0],a[:,1:],axis=0,bounds_error=True)(times)-interp1d(b[:,0],b[:,1:],axis=0,bounds_error=True)(times))
    n=len(xy);temperature=float(errors[:,:n].max());moisture=float(errors[:,n:].max())
    def budget(case,state,label):
        if evidence_dir is None:return assess_dumux_ledger(case/'conservation.csv',duration_s,target_required=state['parameters']['Target.Enabled']=='true')
        # Snapshot complete, actually solved rows through the first state at or
        # after the comparison endpoint. This also supports a running longer
        # reference without inventing a ledger row at an interpolated time.
        data=(case/'conservation.csv').read_bytes();data=data[:data.rfind(b'\n')+1]
        rows=list(csv.reader(io.StringIO(data.decode())));end=rows[0].index('time_s');selected=[rows[0]]
        for row in rows[1:]:
            selected.append(row)
            if float(row[end])>=duration_s:break
        snapshot=evidence_dir/(label+'-ledger.snapshot.csv')
        with snapshot.open('w',newline='') as stream:csv.writer(stream,lineterminator='\n').writerows(selected)
        actual=float(selected[-1][end]);value=assess_dumux_ledger(snapshot,actual,target_required=state['parameters']['Target.Enabled']=='true')
        value.update(original_ledger_path=str(case/'conservation.csv'),comparison_endpoint_covered=actual>=duration_s,snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        if actual<duration_s:value['status']='incomplete'
        return value
    candidate_budget=budget(candidate,new,'candidate');reference_budget=budget(reference,old,'reference')
    result=dict(status='passed' if covers and same_input and same_parameters and temperature<=1e-4 and moisture<=1e-6 and candidate_budget['status']=='passed' and reference_budget['status']=='passed' else 'failed',scope='Native parallel-solver prefix verification only; physical validation and scene admission remain separate',reference=str(reference),candidate=str(candidate),physical_duration_s=duration_s,coverage_s=common_end,same_frozen_input=same_input,same_physical_parameters=same_parameters,temperature_max_difference_k=temperature,temperature_tolerance_k=1e-4,moisture_max_difference=moisture,moisture_tolerance=1e-6,candidate_budget=candidate_budget,reference_budget=reference_budget,source_hashes=dict(reference=old['source_hashes'],candidate=new['source_hashes']),research_accepted=False)
    for label,case in [('reference',reference),('candidate',candidate)]:
        with (case/'surface-frames.bin').open('rb') as stream:result[label+'_surface_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--duration',type=float,default=600.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=verify(a.reference,a.candidate,a.duration,a.output.parent);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
