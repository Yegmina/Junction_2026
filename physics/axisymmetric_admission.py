#!/usr/bin/env python3
"""Durable strict admission for the isolated, native rotational DuMux pair."""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import interp1d

from aerorozum.physics import assess_dumux_ledger
from graded_mesh import axis_nodes
from admission import physical_match
from radial_projection import reconstruction,plan as projection_plan
from run_axisymmetric import convert


def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(case,require_complete=True):
    state=read(case/'run-status.json')
    if require_complete and state['status']!='completed' and state.get('native_status')!='completed':raise ValueError('Required native case is not complete')
    count=read(case/'surface-format.json')['points'];ncol=1+2*count
    path=case/'surface-frames.bin';size=path.stat().st_size//(ncol*8)*ncol
    raw=np.fromfile(path,dtype='<f8',count=size).reshape(-1,ncol)
    if not np.all(np.isfinite(raw)) or np.any(np.diff(raw[:,0])<=0):raise ValueError('Nonfinite/nonmonotone native states')
    xy=np.loadtxt(case/'surface-grid.csv',delimiter=',',skiprows=1).reshape(-1,2)
    p=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',';'));p.optionxform=str
    frozen=case/'input.used.input';private=case/'input.truth.json'
    text=frozen.read_text() if frozen.exists() else (read(private)['input_text'] if private.exists() else Path(__file__).with_name('dumux_smoke.input').read_text())
    p.read_string(text)
    params={s+'.'+k:v for s in p.sections() for k,v in p.items(s)};params.update(state['parameters'])
    return dict(case=case,state=state,params=params,t=raw[:,0],xy=xy,T=raw[:,1:1+count],M=raw[:,1+count:],r=xy[:,0],frozen_input_provenance=frozen.exists() or private.exists())


def weights(t):
    dt=np.diff(t);w=np.zeros_like(t);w[:-1]+=dt/2;w[1:]+=dt/2;return w


def compare_radial(base,candidate):
    if candidate['t'][0]>base['t'][0] or candidate['t'][-1]<base['t'][-1]:raise ValueError('Reference does not cover complete base day')
    r=base['r'];results={}
    edges=axis_nodes(base['params'],0);area=np.pi*np.diff(edges**2);w=weights(base['t'])[:,None]*area[None,:];root=np.sqrt(w)
    for key in ['T','M']:
        spatial=reconstruction(candidate['r'],candidate[key])(r)
        value=interp1d(candidate['t'],spatial,axis=0,bounds_error=True)(base['t'])
        error=value-base[key];at=np.unravel_index(np.argmax(np.abs(error)),error.shape)
        denominator=np.linalg.norm(root*(base[key]-base[key][0])) if key=='T' else np.linalg.norm(root*base[key])
        results[key]=dict(max_error=float(np.max(np.abs(error))),relative_error=float(np.linalg.norm(root*error)/max(denominator,1e-12)),max_error_time_s=float(base['t'][at[0]]),max_error_radius_m=float(r[at[1]]),rms_reference=float(denominator/np.sqrt(w.sum())))
    return results


def correspondence(axis,three,offset=0.):
    cart=native(three,require_complete=False);xy=cart['xy'];local=xy-offset
    use=(local[:,0]>=0)&(local[:,0]<=1)&(local[:,1]>=0)&(local[:,1]<=1)
    radii=np.hypot(local[use,0]-.5,local[use,1]-.5)
    if not len(radii) or radii.max()>axis['r'][-1]:raise ValueError('No shared physical Cartesian ROI inside radial coverage')
    times=cart['t'];keep=(times>=axis['t'][0])&(times<=axis['t'][-1]);times=times[keep]
    result=dict(coverage_s=[float(times[0]),float(times[-1])],full_day=bool(times[-1]-times[0]>=86400 and cart['frozen_input_provenance']),matching_physical_inputs=physical_match(axis,cart,'domain'),three_dimensional_source=str(three),reference_status='Finite-resolution3D diagnostic, not outdoor ground truth',physical_roi_m=[offset,offset+1],cartesian_points=int(use.sum()))
    if result['full_day']:result['reference_budget']=assess_dumux_ledger(three/'conservation.csv',86400,target_required=cart['params']['Target.Enabled']=='true')
    for key in ['T','M']:
        spatial=reconstruction(axis['r'],axis[key])(radii)
        predicted=interp1d(axis['t'],spatial,axis=0,bounds_error=True)(times)
        error=np.abs(predicted-cart[key][keep][:,use]);where=np.unravel_index(np.argmax(error),error.shape);result[key+'_max_difference']=float(error.max());result[key+'_max_difference_time_s']=float(times[where[0]]);result[key+'_max_difference_position_m']=[float(v) for v in local[use][where[1]]]
    return result


def assess(root,publish=False,group_name='axisymmetric-qualification-v2'):
    group=root/'cases'/group_name;required=['present','absent','time','absent-time','grid','absent-grid','domain','absent-domain','regularization','correspondence-depth04','correspondence-absent-depth04']
    missing=[name for name in required if not (group/name/'run-status.json').exists() or (read(group/name/'run-status.json')['status']!='completed' and read(group/name/'run-status.json').get('native_status')!='completed')]
    if missing:
        partial={}
        for name,prefix in [('present',''),('absent','absent-')]:
            try:
                b=native(group/name,False);c=native(group/(prefix+'grid'),False);use=b['t']<=c['t'][-1];a=b.copy()
                for key in ['t','T','M']:a[key]=b[key][use]
                if len(a['t'])>1:partial[name]=dict(coverage_s=float(a['t'][-1]),comparison=compare_radial(a,c),accepted=False)
            except Exception as error:partial[name]=dict(status='pending',reason=str(error))
        return dict(status='waiting_for_full_axisymmetric_references',accepted=False,research_accepted=False,missing=missing,partial_grid_checks=partial)
    cases={name:native(group/name) for name in required};checks=[]
    def add(name,passed,**evidence):checks.append(dict(name=name,status='passed' if passed else 'failed',**evidence))
    upstream=read(root/'upstream/status.json');analytical=read(root/'upstream/analytical-status.json')
    rotation=root/'upstream/rotation-upstream-test.log'
    add('pinned_upstream_regression',upstream.get('status')=='upstream_test_passed' and upstream.get('version')=='3.10.0')
    add('upstream_analytical_conduction_and_water',analytical.get('status')=='upstream_analytical_regressions_passed')
    add('native_rotational_geometry_regression',rotation.exists() and '100% tests passed' in rotation.read_text())
    source=cases['present']['state']['source_hashes']
    add('same_pinned_axisymmetric_application',all(c['state']['backend']=='dumux_axisymmetric_extended_richards_ni' and c['state']['source_hashes']==source for c in cases.values()))
    for name,c in cases.items():
        add('complete_physical_day_'+name,c['t'][0]==0 and c['t'][-1]>=86400)
        add('unfrozen_valid_state_'+name,bool(c['T'].min()>=273.15 and c['T'].max()<370 and c['M'].min()>=0 and c['M'].max()<=1))
        budget=assess_dumux_ledger(c['case']/'conservation.csv',86400,target_required=c['params']['Target.Enabled']=='true')
        add('actual_extruded_DuMux_budget_'+name,budget['status']=='passed',evidence=budget)
    projections={name:projection_plan(cases[name]['r'],cases[name]['T'],cases[name]['M'],maximum_cells=100000)[3] for name in ['present','absent']}
    for name,projection in projections.items():add('cartesian_projection_within_resource_cap_'+name,projection['cartesian_cells_per_axis']<=512,evidence=projection)
    comparisons={}
    for present in [True,False]:
        name='present' if present else 'absent';base=cases[name];prefix='' if present else 'absent-';project=projections[name]
        for kind in ['time','grid','domain',*(['regularization'] if present else [])]:
            other=cases[prefix+kind];add('same_physical_inputs_'+name+'_'+kind,physical_match(base,other,kind))
            comparison=compare_radial(base,other);comparisons[name+'_'+kind]=comparison
            add('temperature_absolute_'+name+'_'+kind,comparison['T']['max_error']+project['temperature_error_bound_k']<=.02,evidence=comparison['T'],cartesian_projection_bound_k=project['temperature_error_bound_k'],tolerance_k=.02)
            add('temperature_relative_'+name+'_'+kind,comparison['T']['relative_error']+project['temperature_error_bound_k']/max(comparison['T']['rms_reference'],1e-12)<=.02)
            add('moisture_relative_'+name+'_'+kind,comparison['M']['relative_error']+project['moisture_error_bound']/max(comparison['M']['rms_reference'],1e-12)<=.02)
        tref=cases[prefix+'time']['params'];p=base['params']
        add('initial_and_maximum_dt_halved_'+name,float(tref['TimeLoop.DtInitial'])<=float(p['TimeLoop.DtInitial'])/2 and float(tref['TimeLoop.MaxTimeStepSize'])<=float(p['TimeLoop.MaxTimeStepSize'])/2)
        for axis in [0,1]:
            coarse=axis_nodes(p,axis);fine=axis_nodes(cases[prefix+'grid']['params'],axis)
            add('nested_radial_vertical_refinement_'+name+'_'+str(axis),len(fine)>=2*(len(coarse)-1)+1 and all(np.min(np.abs(fine-v))<1e-10 for v in coarse))
            bigger=axis_nodes(cases[prefix+'domain']['params'],axis);shift=0 if axis==0 else float(cases[prefix+'domain']['params']['Grid.UpperRight'].split()[1])-float(p['Grid.UpperRight'].split()[1])
            add('domain_doubled_preserving_inner_mesh_'+name+'_'+str(axis),bigger[-1]>=2*coarse[-1] and all(np.min(np.abs(bigger-(v+shift)))<1e-10 for v in coarse))
    p=cases['present']['params'];q=cases['regularization']['params'];add('target_regularization_refined_tenfold',float(q['Target.Porosity'])<=float(p['Target.Porosity'])/10*(1+1e-12) and float(q['Target.Permeability'])<=float(p['Target.Permeability'])/10*(1+1e-12))
    correspondence_results={};pending=False
    for name,axis,three,offset in [('object',cases['correspondence-depth04'],root/'cases/three-d-correspondence-v2/object-h4-domain2',.5),('background',cases['correspondence-absent-depth04'],root/'cases/qualified-skin2-full/absent-grid',0.)]:
        try:
            value=correspondence(axis,three,offset);correspondence_results[name]=value
            if not value['full_day']:pending=True
            add('actual_3d_temperature_correspondence_'+name,value['full_day'] and value['matching_physical_inputs'] and value.get('reference_budget',{}).get('status')=='passed' and value['T_max_difference']<=.02,evidence=value)
        except Exception as error:pending=True;correspondence_results[name]=dict(status='pending',reason=str(error))
    passed=all(x['status']=='passed' for x in checks)
    corepass=all(x['status']=='passed' for x in checks if not x['name'].startswith('actual_3d_'))
    report=dict(status='numerically_accepted_pair' if passed else ('axisymmetric_native_admission_failed' if not corepass else ('waiting_for_full_3d_correspondence' if pending else 'axisymmetric_3d_correspondence_failed')),accepted=passed,research_accepted=passed,backend='dumux_axisymmetric_extended_richards_ni',symmetry_scope='One vertical cylinder, homogeneous flat soil and spatially uniform forcing only',field_validation='not_performed',checks=checks,comparisons=comparisons,projection_bounds=projections,correspondence=correspondence_results,failed_checks=[x['name'] for x in checks if x['status']=='failed'],physical_day_s=86400)
    marker=group/'pair-acceptance.json'
    if passed and publish:
        cells=max(projections['present']['cartesian_cells_per_axis'],projections['absent']['cartesian_cells_per_axis'])
        build=read(root/'upstream/campaign-cached-ledger-surface-graded-params-compact-axisymmetric-status.json')
        for name in ['present','absent']:convert(cases[name]['case'],cases[name]['state']['parameters'],build,cartesian_cells=cells)
        paths={name:group/name/'field.npz' for name in ['present','absent']};hashes={name:digest(path) for name,path in paths.items()}
        report.update(campaign_id=group_name+'-reference-20261114',present_path=str(paths['present']),absent_path=str(paths['absent']),present_sha256=hashes['present'],absent_sha256=hashes['absent'])
        for name,path in paths.items():
            meta=read(path.with_suffix('.json'));meta.update(validation_state='numerically_accepted',research_accepted=True,acceptance_report_path=str(marker),field_sha256=hashes[name]);atomic(path.with_suffix('.json'),meta)
        atomic(marker,report)
        registry=root/'registry.json';data=read(registry) if registry.exists() else dict(fields=[]);identifier=report['campaign_id']
        data['fields']=[x for x in data['fields'] if x.get('campaign_id')!=identifier]+[dict(campaign_id=identifier,split='train',field_path=str(paths['present']),counterfactual_path=str(paths['absent']),acceptance_report_path=str(marker),status='accepted',present_sha256=hashes['present'],absent_sha256=hashes['absent'])];atomic(registry,data)
    else:atomic(marker,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'));p.add_argument('--group',default='axisymmetric-qualification-v2');p.add_argument('--watch',type=float,default=30);p.add_argument('--publish',action='store_true');a=p.parse_args()
    while True:
        try:
            result=assess(a.root,a.publish,a.group);print(json.dumps(result),flush=True)
        except Exception as error:result=dict(status='admission_error',accepted=False,error=f'{type(error).__name__}: {error}');print(json.dumps(result),flush=True)
        atomic(a.root/'cases'/a.group/'admission-progress.json',result)
        if result.get('accepted'):return 0
        if not a.watch:return 1
        time.sleep(a.watch)


if __name__=='__main__':main()
