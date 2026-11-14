#!/usr/bin/env python3
"""Run the real custom DuMux application and convert its actual ASCII fields."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from convert_dumux import convert


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--duration',type=float,default=86400)
    p.add_argument('--dt',type=float,default=120)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cells',default='8 8 10')
    p.add_argument('--no-target',action='store_true')
    p.add_argument('--cached',action='store_true')
    p.add_argument('--ledger',action='store_true')
    p.add_argument('--surface',action='store_true')
    p.add_argument('--graded',action='store_true')
    p.add_argument('--mesh-refinement',type=int,default=1)
    p.add_argument('--vertical-refinement',type=int,default=1)
    p.add_argument('--horizontal-refinement',type=int,default=1)
    p.add_argument('--surface-refinement',type=int,default=1)
    p.add_argument('--cached-parameters',action='store_true')
    p.add_argument('--stream-surfaces',action='store_true',help='Retain every surface sample; prune only processed intermediate volumes, keeping hourly/first/latest')
    p.add_argument('--compact-output',action='store_true')
    p.add_argument('--amgx',action='store_true',help='Use optional-adapter binary; enable GPU explicitly with --set AMGX.Enabled=true')
    p.add_argument('--mpi-compact',action='store_true')
    p.add_argument('--mpi-ranks',type=int,default=1)
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    args=p.parse_args();root=args.root
    file_suffix=('-cached' if args.cached else '')+('-ledger' if args.ledger else '')+('-surface' if args.surface else '')+('-graded' if args.graded else '')+('-params' if args.cached_parameters else '')+('-compact' if args.compact_output else '')+('-amgx' if args.amgx else '')
    if args.mpi_compact:file_suffix+='-mpi'
    if args.mpi_ranks<1 or (args.mpi_ranks>1 and not args.mpi_compact):raise ValueError('Native parallel output requires the MPI compact binary')
    build=json.loads((root/('campaign'+file_suffix+'-status.json')).read_text())
    if build.get('status')!='custom_application_built':raise RuntimeError('Custom DuMux application must be built first')
    if args.duration<=0 or args.dt<=0 or args.threads<1:raise ValueError('Positive duration/dt/threads required')
    cells=[int(v) for v in args.cells.split()]
    if len(cells)!=3 or min(cells)<2:raise ValueError('Three cell counts >=2 required')
    run=args.output_dir;run.mkdir(parents=True,exist_ok=True)
    run_lock=(run/'.run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX)
    parameters={'TimeLoop.TEnd':str(args.duration),'TimeLoop.MaxTimeStepSize':str(args.dt),'TimeLoop.DtInitial':str(min(1.,args.dt/30.)),'Grid.Cells':args.cells,'Problem.Name':'campaign','Target.Enabled':'false' if args.no_target else 'true'}
    for item in args.set:
        key,sep,value=item.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]*',key) or '\n' in value:raise ValueError('Expected KEY=VALUE override')
        if key=='Problem.Name':raise ValueError('Output name override is not allowed')
        parameters[key]=value
    if args.mpi_ranks>1 and parameters.get('AMGX.Enabled','false').lower() in ('true','1'):
        raise ValueError('The current AMGX adapter is single-process; use native distributed ISTL for MPI runs')
    if args.graded:
        from graded_mesh import make_mesh
        extent=tuple(float(v) for v in parameters.get('Grid.UpperRight','1 1 .4').split())
        centre=(float(parameters.get('Target.X',.5)),float(parameters.get('Target.Y',.5)))
        parameters.update(make_mesh(extent,centre,float(parameters.get('Target.Radius',.12)),float(parameters.get('Target.Depth',.03)),float(parameters.get('Target.Thickness',.05)),args.mesh_refinement))
        if min(args.vertical_refinement,args.horizontal_refinement)<1:raise ValueError('Positive grid refinement factors required')
        for axis in range(3):
            factor=args.vertical_refinement if axis==2 else args.horizontal_refinement
            key='Grid.Cells'+str(axis);parameters[key]=' '.join(str(int(v)*factor) for v in parameters[key].split())
        if args.surface_refinement<1:raise ValueError('Positive surface refinement factor required')
        skin=parameters['Grid.Cells2'].split();skin[-1]=str(int(skin[-1])*args.surface_refinement);parameters['Grid.Cells2']=' '.join(skin)
    input_path=Path(__file__).with_name('dumux_smoke.input').resolve()
    input_text=input_path.read_text();input_sha256=hashlib.sha256(input_text.encode()).hexdigest()
    if (run/'run-status.json').exists():
        previous=json.loads((run/'run-status.json').read_text())
        prior_input=json.loads((run/'input.truth.json').read_text()).get('input_text') if (run/'input.truth.json').exists() else None
        if previous.get('status')=='completed' and previous.get('parameters')==parameters and previous.get('source_hashes')==build['source_hashes'] and previous.get('mpi_ranks',1)==args.mpi_ranks and prior_input==input_text and (run/'field.npz').is_file() and (run/'field.json').is_file():
            print(json.dumps(dict(status='reused_completed_case',case=str(run),original_state=previous),indent=2));return
        raise ValueError('Directory contains a non-identical/incomplete run; preserve it and choose a fresh directory')
    frozen_input=run/'input.used.input';frozen_input.write_text(input_text)
    command=[build['binary'],str(frozen_input.resolve())]
    for key,value in parameters.items():command.extend(['-'+key,value])
    if args.mpi_ranks>1:command=['mpiexec',*(['--allow-run-as-root'] if os.geteuid()==0 else []),'-n',str(args.mpi_ranks),*command]
    formulation='surface_component_enthalpy_v2' if args.surface else 'legacy_cell_mixture_v1'
    state=dict(status='running',backend='dumux_extended_richards_ni',solver_version='3.10.0',formulation_version=formulation,research_accepted=False,parameters=parameters,source_hashes=build['source_hashes'],started_at=time.time())
    state['input_sha256']=input_sha256
    state['mpi_ranks']=args.mpi_ranks
    state['linear_backend']='AMGX_fp64_in_process' if parameters.get('AMGX.Enabled','false').lower()=='true' else 'ISTL_ILUBiCGSTAB_cpu'
    if state['linear_backend']=='AMGX_fp64_in_process':
        library=Path(parameters.get('AMGX.LibraryPath',''))
        if not library.is_absolute() or not library.is_file():raise ValueError('Reproducible GPU execution requires an existing absolute AMGX.LibraryPath')
        state['linear_library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    status=run/'run-status.json';status.write_text(json.dumps(state,indent=2))
    started=time.monotonic();env=os.environ.copy();env['DUMUX_NUM_THREADS']=str(args.threads)
    collector=None
    if args.stream_surfaces and not args.compact_output:
        from stream_surfaces import SurfaceCollector
        collector=SurfaceCollector(run,prune=True)
    with (run/'solver.log').open('w') as out:
        if collector:
            process=subprocess.Popen(command,cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    collector.drain();time.sleep(.2)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                collector.checkpoint()
                raise
            result=subprocess.CompletedProcess(command,process.returncode)
        else:result=subprocess.run(command,cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT)
    state.update(exit_code=result.returncode,elapsed_wall_s=time.monotonic()-started,finished_at=time.time())
    if result.returncode:
        state['status']='failed';status.write_text(json.dumps(state,indent=2));raise RuntimeError('DuMux failed; inspect solver.log')
    target_enabled=parameters['Target.Enabled'].lower() in ('true','1')
    targets=[] if not target_enabled else [dict(id='target-001',x_m=float(parameters.get('Target.X',.5)),y_m=float(parameters.get('Target.Y',.5)),radius_m=float(parameters.get('Target.Radius',.12)),depth_m=float(parameters.get('Target.Depth',.03)),thickness_m=float(parameters.get('Target.Thickness',.05)),material='effective_inert_polymer_surrogate')]
    private=run/'input.truth.json';private.write_text(json.dumps(dict(targets=targets,parameters=parameters,input_text=input_text),indent=2))
    if args.compact_output:
        from convert_compact import convert_compact
        fields=convert_compact(run,run/'field.npz',private)
    else:fields=collector.finish(private) if collector else convert(run/'campaign.pvd',run/'field.npz',private)
    meta_path=run/'field.json';meta=json.loads(meta_path.read_text())
    meta.update(config_hash=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()+input_text.encode()).hexdigest(),application_source_hashes=build['source_hashes'],formulation_version=formulation,forcing_start_hour=float(parameters.get('Weather.StartHour',6.)))
    meta['linear_solver_backend']=state['linear_backend']
    meta['mpi_ranks']=args.mpi_ranks
    if 'linear_library_sha256' in state:meta['linear_solver_library_sha256']=state['linear_library_sha256']
    meta_path.write_text(json.dumps(meta,indent=2))
    if args.ledger:
        from aerorozum.physics import assess_dumux_ledger
        assessment=assess_dumux_ledger(run/'conservation.csv',args.duration,target_required=target_enabled)
        (run/'budget-verification.json').write_text(json.dumps(assessment,indent=2))
        state['budget_verification']=assessment
    state.update(status='completed',field_path=str(run/'field.npz'),frames=fields['frames'],shape=fields['shape'])
    status.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))


if __name__=='__main__':main()
