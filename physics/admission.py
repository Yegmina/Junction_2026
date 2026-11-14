#!/usr/bin/env python3
"""Numerical admission for a direct DuMux field from executed evidence only.

No physical field-validation or fast-surrogate acceptance is implied. All
required refinements must cover a full day and use the same physical scene.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np

from aerorozum.physics import assess_dumux_ledger
try:
    from .verify_dumux_cases import compare
    from .graded_mesh import axis_nodes
except ImportError:
    from verify_dumux_cases import compare
    from graded_mesh import axis_nodes


def load_case(path):
    path=Path(path)
    metadata=json.loads(path.with_suffix('.json').read_text())
    truth=json.loads(path.with_suffix('.truth.json').read_text())
    parser=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',';'))
    parser.optionxform=str;parser.read_string(truth['input_text'])
    params={section+'.'+key:value for section in parser.sections() for key,value in parser.items(section)}
    params.update(truth.get('parameters',{}))
    with np.load(path) as data:
        start=float(data['times_s'][0]);end=float(data['times_s'][-1])
        temperature_range=[float(np.min(data['temperature_k'])),float(np.max(data['temperature_k']))]
        moisture_range=[float(np.min(data['moisture'])),float(np.max(data['moisture']))]
    return dict(path=path,metadata=metadata,truth=truth,params=params,start=start,end=end,temperature_range=temperature_range,moisture_range=moisture_range)


def physical_match(base,candidate,kind):
    a=base['params'].copy();b=candidate['params'].copy()
    ignored={'Problem.Name','TimeLoop.MaxTimeStepSize','TimeLoop.DtInitial','Grid.Cells','TimeLoop.TEnd'}
    # Backend selection and matrix diagnostics leave the physical problem
    # unchanged; source matching and executed residual gates remain required.
    ignored|={k for k in set(a)|set(b) if k.startswith('AMGX.')}
    ignored|={k for k in set(a)|set(b) if k.startswith('Grid.Cells')}
    if kind=='regularization':ignored|={'Target.Porosity','Target.Permeability'}
    if kind=='domain':ignored|={'Grid.UpperRight','Target.X','Target.Y'}|{k for k in set(a)|set(b) if k.startswith('Grid.Positions') or k.startswith('Grid.Grading')}
    for key in ignored:a.pop(key,None);b.pop(key,None)
    # Compare numeric spellings semantically (30 == 30.0).
    def canonical(value):
        try:return tuple(float(v) for v in str(value).split())
        except ValueError:return str(value).lower()
    return {k:canonical(v) for k,v in a.items()}=={k:canonical(v) for k,v in b.items()}


def certify_direct(field,time_refined,grid_refined,domain_refined,regularized,upstream_root,output,publish=False):
    paths=dict(base=field,time=time_refined,grid=grid_refined,domain=domain_refined,regularization=regularized)
    cases={name:load_case(path) for name,path in paths.items()}
    base=cases['base'];checks=[]
    def add(name,passed,**evidence):checks.append(dict(name=name,status='not_applicable' if passed is None else ('passed' if passed else 'failed'),**evidence))
    upstream_root=Path(upstream_root)
    upstream=json.loads((upstream_root/'status.json').read_text())
    analytical=json.loads((upstream_root/'analytical-status.json').read_text())
    add('pinned_upstream_regression',upstream.get('status')=='upstream_test_passed' and upstream.get('version')=='3.10.0',evidence=str(upstream_root/'status.json'))
    add('analytical_heat_water',analytical.get('status')=='upstream_analytical_regressions_passed' and analytical.get('conduction_rms_relative_to_temperature_span',float('inf'))<=.01 and analytical.get('richards_manufactured_solution_max_l2',float('inf')) is not None and analytical.get('richards_manufactured_solution_max_l2',float('inf'))<=.001,evidence=analytical)
    add('full_day_reference_coverage',all(c['start']==base['start'] and c['end']>=base['end'] for c in cases.values()) and base['end']-base['start']>=86400,duration_s=base['end']-base['start'])
    source=base['metadata'].get('application_source_hashes')
    add('same_verified_formulation',bool(source) and all(c['metadata'].get('backend')=='dumux_extended_richards_ni' and c['metadata'].get('solver_version')=='3.10.0' and c['metadata'].get('formulation_version')=='surface_component_enthalpy_v2' and c['metadata'].get('temperature_representation')=='reconstructed_boundary_face' and c['metadata'].get('application_source_hashes')==source for c in cases.values()))
    for name,c in cases.items():
        p=c['params']
        valid=(0<float(p['SpatialParams.Porosity'])<1 and float(p['SpatialParams.Permeability'])>0 and 0<=float(p['Weather.RelativeHumidity'])<=1 and 0<float(p['Weather.Emissivity'])<=1 and 0<=float(p['Weather.Albedo'])<=1 and float(p['Weather.Wind'])>=0)
        add('physical_parameter_envelope_'+name,valid)
        add('unfrozen_model_domain_'+name,c['temperature_range'][0]>=273.15 and c['temperature_range'][1]<370 and c['moisture_range'][0]>=0 and c['moisture_range'][1]<=1,temperature_range_k=c['temperature_range'],moisture_range=c['moisture_range'],reason='This liquid-water model does not simulate freezing/thawing')
        if name!='base':add('physical_scene_matches_'+name,physical_match(base,c,name))
        ledger=assess_dumux_ledger(c['path'].parent/'conservation.csv',c['end'],target_required=bool(c['truth'].get('targets')))
        add('actual_dumux_budgets_'+name,ledger['status']=='passed',evidence=ledger)
    params=base['params'];nodes=[axis_nodes(params,axis) for axis in range(3)]
    fine_nodes=[axis_nodes(cases['grid']['params'],axis) for axis in range(3)]
    add('time_step_refined_by_two',float(cases['time']['params']['TimeLoop.MaxTimeStepSize'])<=float(params['TimeLoop.MaxTimeStepSize'])/2)
    add('initial_time_step_refined_by_two',float(cases['time']['params']['TimeLoop.DtInitial'])<=float(params['TimeLoop.DtInitial'])/2)
    add('time_and_material_screens_preserve_grid',all(np.array_equal(axis_nodes(cases[name]['params'],axis),nodes[axis]) for name in ['time','regularization'] for axis in range(3)))
    add('references_do_not_coarsen_time',all(float(c['params']['TimeLoop.MaxTimeStepSize'])<=float(params['TimeLoop.MaxTimeStepSize']) for c in cases.values()))
    refined=all(len(fine)>=2*(len(coarse)-1)+1 and all(np.min(np.abs(fine-v))<1e-10 for v in coarse) for coarse,fine in zip(nodes,fine_nodes))
    add('grid_refined_by_two',refined)
    has_target=bool(base['truth'].get('targets'))
    add('target_regularization_refined_by_ten',(float(cases['regularization']['params']['Target.Porosity'])<=float(params['Target.Porosity'])/10*(1+1e-12) and float(cases['regularization']['params']['Target.Permeability'])<=float(params['Target.Permeability'])/10*(1+1e-12)) if has_target else None,reason='Not applicable for an object-free counterpart' if not has_target else 'Executed lower-porosity/permeability target')
    extent=np.asarray([float(v) for v in params['Grid.UpperRight'].split()]);larger=np.asarray([float(v) for v in cases['domain']['params']['Grid.UpperRight'].split()])
    add('domain_enlarged_by_two',bool(np.all(larger>=2*extent)))
    shift=(larger[:2]-extent[:2])/2
    shifts=[*shift,larger[2]-extent[2]]
    domain_nodes=[axis_nodes(cases['domain']['params'],axis) for axis in range(3)]
    preserved=all(all(np.min(np.abs(domain_nodes[axis]-(value+shifts[axis])))<1e-10 for value in nodes[axis]) for axis in range(3))
    add('domain_screen_preserves_original_grid_nodes',preserved)
    centred=all(abs(float(cases['domain']['params'][k])-float(params[k])-shift[i])<1e-9 for i,k in enumerate(['Target.X','Target.Y']))
    add('domain_target_translation_matches',centred)
    if abs(shift[0]-shift[1])>1e-9:raise ValueError('This verifier requires equal x/y domain extensions')
    for name in ['time','grid','regularization','domain']:
        if name=='regularization' and not has_target:
            add('temperature_convergence_regularization',None,reason='Object-free counterpart has no regularized target')
            continue
        comparison=compare(base['path'],cases[name]['path'],coordinate_shift_m=float(shift[0]) if name=='domain' else 0.)
        add('temperature_convergence_'+name,comparison['status']=='passed',evidence=comparison)
        add('temperature_relative_convergence_'+name,comparison['temperature_relative_error']<=.02,relative_error=comparison['temperature_relative_error'],tolerance=.02)
        add('moisture_relative_convergence_'+name,comparison['moisture_relative_error']<=.02,relative_error=comparison['moisture_relative_error'],tolerance=.02)
    accepted=all(c['status'] in ('passed','not_applicable') for c in checks)
    hashes={name:hashlib.sha256(c['path'].read_bytes()).hexdigest() for name,c in cases.items()}
    report=dict(status='numerically_accepted' if accepted else 'numerical_admission_failed',accepted=accepted,research_accepted=accepted,backend='dumux_extended_richards_ni',formulation_version='surface_component_enthalpy_v2',field_path=str(base['path']),field_hashes=hashes,checks=checks,field_validation='not_performed',published_experiment_reproduction='not_performed',surrogate_admission='separate_additional_gate',meaning='Numerical acceptance of this direct field and its specified evidence, not a physical calibration or a claim of confirmed mines')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    if publish and accepted:
        meta=base['metadata'];meta.update(validation_state='numerically_accepted',research_accepted=True,acceptance_report_path=str(output),field_sha256=hashes['base'],field_validation='not_performed')
        base['path'].with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['field','time-refined','grid-refined','domain-refined','regularized','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--upstream-root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'))
    p.add_argument('--publish',action='store_true')
    a=p.parse_args();r=certify_direct(a.field,a.time_refined,a.grid_refined,a.domain_refined,a.regularized,a.upstream_root,a.output,a.publish);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
