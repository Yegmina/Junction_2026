#!/usr/bin/env python3
"""Execute paired native/GPU DuMux cases and require actual physics parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .verify_dumux_cases import compare
except ImportError:
    from verify_dumux_cases import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--duration',type=float,default=600)
    p.add_argument('--mesh-refinement',type=int,default=1)
    p.add_argument('--native-case',type=Path,help='Reuse an already completed matching CPU benchmark only')
    p.add_argument('--configuration',help='Explicit AMGX JSON configuration override')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    runner=Path(__file__).with_name('run_dumux.py')
    shared=[str(runner),'--root',str(args.root/'upstream'),'--duration',str(args.duration),'--dt','30','--threads',str(args.threads),'--cached','--ledger','--surface','--graded','--cached-parameters','--compact-output','--amgx','--surface-refinement','2','--mesh-refinement',str(args.mesh_refinement)]
    cases={};states={}
    for name,gpu in [('native',False),('gpu',True)]:
        case=args.native_case if name=='native' and args.native_case else out/name
        cases[name]=case
        status=case/'run-status.json'
        if not status.exists():
            command=[sys.executable,*shared,'--output-dir',str(case),'--set','AMGX.Enabled='+str(gpu).lower(),'--set','AMGX.ExportMatrix=true']
            if gpu:
                command+=['--set','AMGX.LibraryPath='+str(args.library)]
                if args.configuration:command+=['--set','AMGX.Configuration='+args.configuration]
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
            with (out/(name+'-launcher.log')).open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                report=dict(status='failed',reason='Actual DuMux process failed',case=str(case),exit_code=result.returncode,accepted=False)
                (out/'gpu-backend-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return 1
        state=json.loads(status.read_text());states[name]=state
        expected=shared[shared.index('--duration')+1]
        if state.get('status')!='completed' or float(state['parameters']['TimeLoop.TEnd'])!=float(expected):raise ValueError('Matching completed physics case required')
        from graded_mesh import make_mesh
        expected_cells=make_mesh(refinement=args.mesh_refinement)
        # Surface refinement doubles only the top segment count.
        top=expected_cells['Grid.Cells2'].split();top[-1]=str(2*int(top[-1]));expected_cells['Grid.Cells2']=' '.join(top)
        if any(state['parameters'].get(k)!=v for k,v in expected_cells.items()):raise ValueError('CPU/GPU benchmark mesh mismatch')
    parity=compare(cases['native']/'field.npz',cases['gpu']/'field.npz',tolerance_k=.0001)
    solves=[json.loads(line) for line in (cases['gpu']/'gpu-linear-solves.jsonl').read_text().splitlines() if line]
    budgets={name:json.loads((case/'budget-verification.json').read_text()) for name,case in cases.items()}
    native=json.loads((cases['native']/'native-linear-baseline.json').read_text())
    passed=parity['status']=='passed' and solves and all(x.get('passed') for x in solves) and all(x.get('status')=='passed' for x in budgets.values())
    before=states['native']['elapsed_wall_s'];after=states['gpu']['elapsed_wall_s']
    report=dict(status='passed' if passed else 'failed',accepted=False,meaning='GPU linear-backend equivalence only; full-day numerical admission remains separate',parity=parity,budgets=budgets,gpu_solve_calls=len(solves),native_linear_reference=native,native_wall_s=before,gpu_wall_s=after,end_to_end_speed_ratio=before/after,runtime_improved=after<before,gpu_linear_warm_mean_s=sum(sum(x[k] for k in ['preparation_s','setup_upload_s','solve_download_s']) for x in solves[1:])/max(len(solves)-1,1),timing_context='Sequential cases on the same VM; full case timing includes GPU initialization, transfers, host assembly and output',source_hashes_match=states['native']['source_hashes']==states['gpu']['source_hashes'])
    if not report['source_hashes_match']:report['status']='failed'
    (out/'gpu-backend-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    (cases['gpu']/'gpu-backend-parity.json').write_text(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
