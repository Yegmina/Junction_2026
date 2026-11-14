#!/usr/bin/env python3
"""Start a bounded genuine large-mesh AMGX case on an explicitly reserved GPU.

The caller coordinates GPU ownership with the training controller first. This
script provisions nothing, changes no physical coefficients, and grants no
numerical acceptance. All residual checks in the native adapter remain active.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'));p.add_argument('--device',type=int,required=True);p.add_argument('--unit',default='aerorozum-physics-amgx-large-benchmark');p.add_argument('--duration',type=float,default=600);p.add_argument('--wall-limit',type=int,default=600);a=p.parse_args()
    if a.device not in range(4) or not 1<=a.wall_limit<=600:raise ValueError('Existing worker GPU index and <=10min wall guard required')
    root=a.root;case=root/'cases/amgx-large-benchmark-20261114'
    if case.exists():raise ValueError('Preserve existing evidence; do not overwrite an earlier GPU case')
    config=dict(config_version=2,solver=dict(solver='BICGSTAB',max_iters=5000,tolerance=1e-12,norm='L2',convergence='RELATIVE_INI_CORE',monitor_residual=1,print_solve_stats=0,preconditioner=dict(solver='MULTICOLOR_DILU',max_iters=1)))
    command=['/opt/aerorozum/venv/bin/python',str(root/'source/physics/run_dumux.py'),'--root',str(root/'upstream'),'--output-dir',str(case),'--duration',str(a.duration),'--dt','30','--threads','8','--cached','--ledger','--surface','--graded','--cached-parameters','--compact-output','--amgx','--surface-refinement','2','--horizontal-refinement','4','--vertical-refinement','2','--set','Grid.UpperRight=2 2 .4','--set','Target.X=1','--set','Target.Y=1','--set','AMGX.Enabled=true','--set','AMGX.LibraryPath=/opt/aerorozum/gpu-physics/AMGX/build/libamgxsh.so','--set','AMGX.Configuration='+json.dumps(config,separators=(',',':'))]
    subprocess.run(['systemd-run','--unit='+a.unit,'--setenv=PYTHONPATH=/opt/aerorozum/physics/source','--setenv=CUDA_VISIBLE_DEVICES='+str(a.device),'--property=CPUQuota=800%','--property=MemoryMax=24G','--property=RuntimeMaxSec='+str(a.wall_limit),*command],check=True)
    print(json.dumps(dict(status='launched',case=str(case),unit=a.unit,gpu=a.device,wall_limit_s=a.wall_limit,physical_duration_s=a.duration,research_accepted=False)))


if __name__=='__main__':main()
