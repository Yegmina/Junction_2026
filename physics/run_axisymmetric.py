#!/usr/bin/env python3
"""Run native DuMux rotational Richards/heat, never an accepted 3D shortcut.

Scope: one vertical cylindrical inclusion, flat homogeneous soil and spatially
uniform atmospheric forcing. Native coupled equations use 2*pi*r geometry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    from .radial_projection import plan as projection_plan,frame as projection_frame
except ImportError:
    from radial_projection import plan as projection_plan,frame as projection_frame


def mesh(radius=1.,depth=.8,target_radius=.12,target_depth=.03,thickness=.05,refinement=1):
    if radius<=target_radius+4*target_depth or refinement<1:raise ValueError('Resolved target must be contained in the radial domain')
    anchors=sorted(set([0.,radius,target_radius,*[v for v in [target_radius/2,target_radius+target_depth,target_radius+2*target_depth,target_radius+4*target_depth,target_radius+8*target_depth,.5,1.,2.,4.,8.] if 0<v<radius]]))
    counts=[]
    for a,b in zip(anchors,anchors[1:]):
        spacing=min(.005,target_depth/6) if (a+b)/2<target_radius+2*target_depth else (.01 if (a+b)/2<target_radius+4*target_depth else .05)
        counts.append(max(1,int(np.ceil((b-a)/spacing-1e-9)))*refinement)
    positions=sorted(set([0.,depth,*[v for v in [depth-.1,depth-.2,depth-.4,depth-.8,depth-1.6,depth-3.2,depth-target_depth-thickness,depth-target_depth] if 0<v<depth]]))
    vertical=[]
    for lower,upper in zip(positions,positions[1:]):
        below=depth-(lower+upper)/2;spacing=.0005 if below<target_depth else (.0025 if below<target_depth+thickness else (.005 if below<.1 else (.01 if below<.2 else (.025 if below<.4 else .05))))
        vertical.append(max(1,int(np.ceil((upper-lower)/spacing-1e-9)))*refinement)
    return {'Grid.UpperRight':f'{radius:.12g} {depth:.12g}','Grid.Positions0':' '.join(format(v,'.12g') for v in anchors),'Grid.Cells0':' '.join(map(str,counts)),'Grid.Positions1':' '.join(format(v,'.12g') for v in positions),'Grid.Cells1':' '.join(map(str,vertical))}


def convert(case,parameters,build,physical_width=1.,cartesian_cells=100):
    description=json.loads((case/'surface-format.json').read_text());count=int(description['points']);columns=1+2*count
    coordinates=np.loadtxt(case/'surface-grid.csv',delimiter=',',skiprows=1).reshape(-1,2)
    r=coordinates[:,0];raw=np.fromfile(case/'surface-frames.bin',dtype='<f8')
    if len(raw)%columns:raise ValueError('Incomplete native radial surface record')
    records=raw.reshape(-1,columns)
    ledger=np.atleast_1d(np.genfromtxt(case/'conservation.csv',delimiter=',',names=True))
    if len(ledger)!=len(records)-1 or not np.allclose(ledger['time_s'],records[1:,0],atol=1e-8,rtol=0):raise ValueError('Native radial output does not cover every solved physical state')
    if not np.all(np.diff(r)>0) or not np.all(np.isfinite(records)):raise ValueError('Invalid native radial field')
    cartesian_cells,tp,mp,projection=projection_plan(r,records[:,1:count+1],records[:,count+1:],width=physical_width,minimum_cells=cartesian_cells)
    x=(np.arange(cartesian_cells)+.5)*physical_width/cartesian_cells;y=x.copy();xx,yy=np.meshgrid(x,y)
    centre=physical_width/2;query=np.hypot(xx-centre,yy-centre)
    if query.max()>r[-1]:raise ValueError('Cartesian patch extends beyond native radial field; no extrapolation')
    temperature=np.empty((len(records),len(y),len(x)),dtype='float32');moisture=np.empty_like(temperature)
    for i,row in enumerate(records):
        # First annulus is constant towards the symmetry axis; no values are
        # inferred outside the physical radial domain.
        temperature[i]=projection_frame(tp,i,query);moisture[i]=projection_frame(mp,i,query)
    output=case/'field.npz'
    temporary=case/'field.projecting.tmp.npz'
    np.savez_compressed(temporary,times_s=records[:,0],x_m=x,y_m=y,temperature_k=temperature,moisture=moisture,r_m=r,radial_temperature_k=records[:,1:count+1],radial_moisture=records[:,count+1:]);temporary.replace(output)
    enabled=parameters['Target.Enabled']=='true'
    truth=dict(targets=[] if not enabled else [dict(id='target-001',x_m=centre,y_m=centre,radius_m=float(parameters.get('Target.Radius',.12)),depth_m=float(parameters.get('Target.Depth',.03)),thickness_m=float(parameters.get('Target.Thickness',.05)),material='effective_inert_polymer_surrogate')],parameters=parameters,input_text=(case/'input.used.input').read_text())
    output.with_suffix('.truth.json').write_text(json.dumps(truth,indent=2))
    metadata=dict(backend='dumux_axisymmetric_extended_richards_ni',solver_version='3.10.0',fidelity='experimental_native_coupled_axisymmetric_DuMux',formulation_version='axisymmetric_surface_component_enthalpy_v1',validation_state='axisymmetric_candidate_unverified',research_accepted=False,field_validation='not_performed',temperature_representation='revolved_reconstructed_boundary_face',application_source_hashes=build['source_hashes'],config_hash=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()+(case/'input.used.input').read_bytes()).hexdigest(),units=dict(times_s='s',x_m='m east',y_m='m north',temperature_k='K',moisture='m3 water / m3 bulk',r_m='m from symmetry axis'),native_radial_arrays=['r_m','radial_temperature_k','radial_moisture'],native_surface_sha256=hashlib.sha256((case/'surface-frames.bin').read_bytes()).hexdigest(),axisymmetric_scope='Single vertical cylinder, homogeneous flat soil, spatially uniform forcing; excludes multiple objects, heterogeneous soil, slopes and directional shading',coordinate_projection='Native radial profile linearly interpolated at Cartesian cell centres; projection admission remains separate',physical_domain=dict(radial_radius_m=float(parameters['Grid.UpperRight'].split()[0]),depth_m=float(parameters['Grid.UpperRight'].split()[1]),cartesian_patch_width_m=physical_width),forcing_start_hour=float(parameters.get('Weather.StartHour',6.)))
    metadata.update(cartesian_projection=projection,native_reconstruction=projection['native_reconstruction'],coordinate_projection='Bounded bilinear Cartesian representation of the C1 native radial reconstruction')
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2));return dict(field_path=str(output),frames=len(records),native_radial_cells=count,cartesian_shape=list(temperature.shape))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'));p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--duration',type=float,default=86400);p.add_argument('--dt',type=float,default=30);p.add_argument('--initial-dt',type=float,default=1);p.add_argument('--radius',type=float,default=1);p.add_argument('--depth',type=float,default=.8);p.add_argument('--refinement',type=int,default=1);p.add_argument('--threads',type=int,default=1);p.add_argument('--no-target',action='store_true');p.add_argument('--set',action='append',default=[]);p.add_argument('--cartesian-cells',type=int,default=100)
    a=p.parse_args();manifest=a.root/'campaign-cached-ledger-surface-graded-params-compact-axisymmetric-status.json';build=json.loads(manifest.read_text());case=a.output_dir;case.mkdir(parents=True,exist_ok=True)
    if (case/'run-status.json').exists():raise ValueError('Preserve existing case and use a fresh output directory')
    params={'TimeLoop.TEnd':str(a.duration),'TimeLoop.MaxTimeStepSize':str(a.dt),'TimeLoop.DtInitial':str(a.initial_dt),'Problem.Name':'campaign','Target.Enabled':'false' if a.no_target else 'true'}
    for item in a.set:
        key,sep,value=item.partition('=')
        if not sep or '\n' in value or key=='Problem.Name':raise ValueError('Expected safe KEY=VALUE physics override')
        params[key]=value
    params.update(mesh(a.radius,a.depth,float(params.get('Target.Radius',.12)),float(params.get('Target.Depth',.03)),float(params.get('Target.Thickness',.05)),a.refinement))
    text=Path(__file__).with_name('dumux_smoke.input').read_text();frozen=case/'input.used.input';frozen.write_text(text)
    command=[build['binary'],str(frozen.resolve())]
    for key,value in params.items():command+=['-'+key,value]
    state=dict(status='running',backend='dumux_axisymmetric_extended_richards_ni',solver_version='3.10.0',research_accepted=False,parameters=params,source_hashes=build['source_hashes'],started_at=time.time(),input_sha256=hashlib.sha256(text.encode()).hexdigest())
    state['mesh_strategy']='axisymmetric_deep_heat_resolution_v2'
    status=case/'run-status.json';status.write_text(json.dumps(state,indent=2));env=os.environ.copy();env['DUMUX_NUM_THREADS']=str(a.threads);began=time.monotonic()
    with (case/'solver.log').open('w') as log:result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
    state.update(exit_code=result.returncode,elapsed_wall_s=time.monotonic()-began,finished_at=time.time())
    if result.returncode:state['status']='failed';status.write_text(json.dumps(state,indent=2));raise RuntimeError('Native axisymmetric solver failed; inspect solver.log')
    state.update(native_status='completed',status='projecting');status.write_text(json.dumps(state,indent=2))
    try:fields=convert(case,params,build,cartesian_cells=a.cartesian_cells)
    except Exception as error:
        state.update(status='native_completed_projection_failed',projection_error=str(error));status.write_text(json.dumps(state,indent=2));raise
    from aerorozum.physics import assess_dumux_ledger
    budget=assess_dumux_ledger(case/'conservation.csv',a.duration,target_required=not a.no_target);budget['backend']='dumux_axisymmetric_extended_richards_ni';(case/'budget-verification.json').write_text(json.dumps(budget,indent=2))
    state.update(status='completed',**fields,budget_verification=budget);status.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))


if __name__=='__main__':main()
