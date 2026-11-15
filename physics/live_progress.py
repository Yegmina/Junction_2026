#!/usr/bin/env python3
"""Publish honest live progress/previews from real DuMux surface checkpoints."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
from scipy.interpolate import RegularGridInterpolator

from graded_mesh import axis_nodes


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def latest_ledger(path):
    try:data=path.read_bytes()
    except FileNotFoundError:return None
    if not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
    rows=list(csv.DictReader(io.StringIO(data.decode())))
    for row in reversed(rows):
        try:
            values={k:float(v) for k,v in row.items()}
            if all(np.isfinite(v) for v in values.values()):return values
        except (ValueError,TypeError):continue
    return None


def latest_surface(case):
    description=read_json(case/'surface-format.json')
    if not description:return None
    count=int(description['points']);size=(1+2*count)*8;path=case/'surface-frames.bin'
    try:
        complete=path.stat().st_size//size
        if not complete:return None
        with path.open('rb') as data:data.seek((complete-1)*size);record=np.frombuffer(data.read(size),dtype='<f8').copy()
    except FileNotFoundError:return None
    if len(record)!=1+2*count or not np.all(np.isfinite(record)):return None
    coordinates=np.genfromtxt(case/'surface-grid.csv',delimiter=',',names=True)
    x=np.unique(coordinates['x_m']);y=np.unique(coordinates['y_m'])
    if len(x)*len(y)!=count:raise ValueError('Preview requires actual tensor-product surface topology')
    return float(record[0]),record[1:count+1].reshape(len(y),len(x)),record[count+1:].reshape(len(y),len(x))


def partial_grid_check(base,fine):
    def history(case):
        description=read_json(case/'surface-format.json');count=int(description['points']);columns=1+2*count
        path=case/'surface-frames.bin';complete=path.stat().st_size//(columns*8)
        raw=np.fromfile(path,dtype='<f8',count=complete*columns).reshape(complete,columns)
        if not np.all(np.isfinite(raw)):raise ValueError('Nonfinite native solver state; convergence cannot be reported passing')
        coordinates=np.round(np.loadtxt(case/'surface-grid.csv',delimiter=',',skiprows=1),9)
        x=np.unique(coordinates[:,0]);y=np.unique(coordinates[:,1])
        return raw[:,0],x,y,raw[:,1:count+1].reshape(-1,len(y),len(x))
    t,x,y,temperature=history(base);tt,xx,yy,fine_temperature=history(fine)
    if len(t)<2 or len(tt)<2:return dict(status='waiting_for_solver_states')
    use=(t>=tt[0])&(t<=tt[-1]);t=t[use];temperature=temperature[use]
    qtime,qy,qx=np.meshgrid(t,y,x,indexing='ij');points=np.stack((qtime,qy,qx),axis=-1)
    reference=RegularGridInterpolator((tt,yy,xx),fine_temperature,bounds_error=True)(points)
    error=np.abs(reference-temperature);where=np.unravel_index(np.argmax(error),error.shape);maximum=float(error[where])
    status='failed_on_partial_coverage' if maximum>.02006 else ('borderline_requires_final_serialization_check' if maximum>.02 else 'passing_so_far_not_accepted')
    return dict(status=status,max_temperature_error_k=maximum,max_error_time_s=float(t[where[0]]),coverage_s=[float(t[0]),float(t[-1])],required_duration_s=86400.,tolerance_k=.02,source='Actual complete native float64 surface checkpoints; fine interpolated at base cell centres/times without extrapolation',accepted=False)


def preview(temperature,params,path,title,virtual_time,accepted=False):
    width=480;height=480
    p={'Grid.UpperRight':'1 1 .4',**params};edges=[axis_nodes(p,axis) for axis in (0,1)]
    xs=edges[0][0]+(np.arange(width)+.5)/width*(edges[0][-1]-edges[0][0]);ys=edges[1][0]+(np.arange(height)+.5)/height*(edges[1][-1]-edges[1][0])
    ix=np.searchsorted(edges[0],xs)-1;iy=np.searchsorted(edges[1],ys)-1
    if temperature.shape!=(len(edges[1])-1,len(edges[0])-1):raise ValueError('Preview geometry differs from actual solver grid')
    raster=temperature[iy[:,None],ix[None,:]][::-1]
    low=float(np.min(temperature));high=float(np.max(temperature));normalized=(raster-low)/max(high-low,1e-9)
    anchors=np.array([[5,4,20],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]],dtype=float)
    palette=np.stack([np.interp(np.arange(256),np.linspace(0,255,len(anchors)),anchors[:,i]) for i in range(3)],axis=-1).astype('uint8')
    colors=palette[np.clip(np.rint(normalized*255),0,255).astype('uint8')]
    image=Image.new('RGB',(640,560),'#111827');image.paste(Image.fromarray(colors),(20,46));draw=ImageDraw.Draw(image)
    label='numerically accepted' if accepted else 'unaccepted preview'
    draw.text((20,12),title[:80],fill='white');draw.text((20,534),f'Actual DuMux surface | t={virtual_time/3600:.2f} h | {label}',fill='white')
    for row in range(height):draw.line([(528,46+row),(555,46+row)],fill=tuple(int(v) for v in palette[255-round(255*row/(height-1))]))
    draw.text((562,46),f'{high:.3f} K',fill='white');draw.text((562,510),f'{low:.3f} K',fill='white');draw.text((515,260),'Autoscale',fill='white')
    temporary=path.with_suffix('.tmp.png');image.save(temporary);temporary.replace(path)
    return [low,high]


def update(root,output,media_dir):
    group=root/'cases/qualified-skin2-full';cases=[];media=[];now=time.time();media_dir.mkdir(parents=True,exist_ok=True)
    groups=[group,root/'cases/three-d-correspondence-v2',*sorted((root/'cases').glob('axisymmetric-qualification*'))];overrides={};entries=[]
    for current in groups:
        for status in current.glob('*status.json'):
            for name,job in (read_json(status,{}) or {}).get('jobs',{}).items():overrides[(current.name,name)]=job
        if current.exists():entries.extend(current.iterdir())
    for case in sorted(entries):
        if not case.is_dir():continue
        state=read_json(case/'run-status.json')
        if not state:continue
        params=state['parameters'];duration=float(params['TimeLoop.TEnd']);ledger=latest_ledger(case/'conservation.csv');surface=latest_surface(case)
        virtual=ledger['time_s'] if ledger else (surface[0] if surface else 0.)
        axisymmetric=state.get('backend')=='dumux_axisymmetric_extended_richards_ni'
        case_id=((case.parent.name.replace('axisymmetric-qualification','axisym')+'-') if axisymmetric else '')+case.name
        status=state['status'];job=overrides.get((case.parent.name,case.name),{})
        if job.get('status') in ('failed','timed_out'):status=job['status']
        elapsed=max(0,now-state['started_at']);eta=elapsed*(duration-virtual)/virtual if virtual>60 and virtual<duration and status=='running' else None
        item=dict(case_id=case_id,status=status,source=state.get('backend','genuine_DuMux_3.10.0_coupled_heat_Richards_vapour'),field_path=str(case/'field.npz'),virtual_time_s=virtual,duration_s=duration,progress=min(1.,virtual/duration),elapsed_wall_s=state.get('elapsed_wall_s',elapsed),remaining_wall_s_estimate=eta,eta_basis='Linear extrapolation of measured virtual-time progress; weather/moisture nonlinear stiffness can change it',research_accepted=False,validation_state='full_numerical_checks_pending',linear_backend=state.get('linear_backend'),source_hashes=state.get('source_hashes'))
        metadata=read_json(case/'field.json',{})
        if metadata.get('research_accepted') and metadata.get('validation_state')=='numerically_accepted':item.update(research_accepted=True,validation_state='numerically_accepted')
        if ledger:item['actual_latest_budget']=ledger
        if surface:
            image=media_dir/(case_id+'.png')
            try:
                display=surface[1];display_params=params
                if axisymmetric:
                    radii=np.loadtxt(case/'surface-grid.csv',delimiter=',',skiprows=1)[:,0]
                    x=(np.arange(160)+.5)/160;xx,yy=np.meshgrid(x,x);rr=np.hypot(xx-.5,yy-.5)
                    if rr.max()>radii[-1]:raise ValueError('Preview would exceed physical radial coverage')
                    display=np.interp(rr,radii,surface[1].ravel());display_params={'Grid.UpperRight':'1 1 .8','Grid.Cells':'160 160 2'}
                item['surface_temperature_range_k']=preview(display,display_params,image,case_id,surface[0],item['research_accepted']);item['preview_time_s']=surface[0]
                media.append(dict(image='pipeline/media/physics/'+image.name,title='Coupled surface temperature: '+case_id,case_id=case_id,virtual_time_s=surface[0],research_accepted=item['research_accepted']))
            except Exception as error:item['preview_error']=f'{type(error).__name__}: {error}'
        cases.append(item)
    registry=read_json(root/'registry.json',{'fields':[]});accepted=[x for x in registry.get('fields',[]) if x.get('status')=='accepted']
    payload=dict(status='running' if any(x['status']=='running' for x in cases) else 'waiting',updated_at=now,source='Actual local DuMux solver ledger and float64 surface checkpoints; no generated progress',cases=cases,accepted_pairs=len(accepted),registry_path=str(root/'registry.json'),media=media,field_validation='not_performed',research_accepted=False)
    payload['partial_grid_checks']={}
    payload['parameter_envelope_qualification']=read_json(Path(__file__).with_name('envelope-plan.json'),dict(status='planned_not_qualified',research_accepted=False))
    payload['controller']=read_json(root/'controller-state.json',{})
    payload['axisymmetric_qualification']={g.name:read_json(g/'admission-progress.json',dict(status='pending',accepted=False)) for g in groups if g.name.startswith('axisymmetric-')}
    for name,a,b in [('present','present','grid'),('absent','absent','absent-grid')]:
        try:payload['partial_grid_checks'][name]=partial_grid_check(group/a,group/b)
        except Exception as error:payload['partial_grid_checks'][name]=dict(status='pending',reason=f'{type(error).__name__}: {error}')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.tmp.json');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(output)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'));p.add_argument('--output',type=Path,default=Path('/opt/aerorozum/artifacts/pipeline/physics.json'));p.add_argument('--media-dir',type=Path,default=Path('/opt/aerorozum/artifacts/pipeline/media/physics'));p.add_argument('--watch',type=float,default=10)
    a=p.parse_args()
    while True:
        try:result=update(a.root,a.output,a.media_dir);print(json.dumps(dict(cases=len(result['cases']),accepted_pairs=result['accepted_pairs'],updated_at=result['updated_at'])),flush=True)
        except Exception as error:print(json.dumps(dict(error=f'{type(error).__name__}: {error}')),flush=True)
        if not a.watch:return
        time.sleep(a.watch)


if __name__=='__main__':main()
