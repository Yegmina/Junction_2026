#!/usr/bin/env python3
"""Bounded parallel physics experiments on the existing authorized VM.

No VMs are provisioned and no training files are touched. A pilot is evidence
for planning full-day convergence runs, never a substitute for those runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/opt/aerorozum/physics'))
    p.add_argument('--stage',choices=['pilot','full'],default='pilot')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--max-wall-seconds',type=float,default=4800)
    p.add_argument('--cached-parameters',action='store_true')
    p.add_argument('--stream-surfaces',action='store_true')
    p.add_argument('--compact-output',action='store_true')
    p.add_argument('--amgx-adapter',action='store_true')
    p.add_argument('--surface-refinement',type=int,default=1)
    p.add_argument('--case-group',help='Explicit fresh shared case directory below cases/')
    p.add_argument('--status-name',default='suite-status.json')
    p.add_argument('--gpu-devices',help='Optional comma-separated physical GPU IDs, one independent process per GPU')
    p.add_argument('--amgx-library',type=Path)
    p.add_argument('--only',action='append',choices=['present','time','grid','domain','regularization','absent','absent-time','absent-grid','absent-domain'])
    args=p.parse_args();root=args.root;duration=600 if args.stage=='pilot' else 86400
    if args.case_group and not re.fullmatch('[a-zA-Z0-9-]+',args.case_group):raise ValueError('Simple case-group name required')
    if not re.fullmatch('[a-zA-Z0-9-]+[.]json',args.status_name):raise ValueError('Simple JSON status filename required')
    if args.surface_refinement<1:raise ValueError('Positive skin refinement required')
    devices=[] if not args.gpu_devices else [int(v) for v in args.gpu_devices.split(',')]
    if devices and (not args.amgx_adapter or not args.amgx_library or len(set(devices))!=len(devices) or min(devices)<0 or args.workers>len(devices)):raise ValueError('GPU runs require adapter/library and <= one worker per distinct GPU')
    tokens=queue.Queue()
    for device in devices:tokens.put(device)
    output=root/'cases'/(args.case_group or ('graded-'+args.stage+('-params' if args.cached_parameters else '')+('-compact' if args.compact_output else '')));output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    settings=[('present',[]),('time',['--dt','15']),('grid',['--mesh-refinement','2']),('domain',['--set','Grid.UpperRight=2 2 .8','--set','Target.X=1','--set','Target.Y=1']),('regularization',['--set','Target.Porosity=1e-7','--set','Target.Permeability=1e-25']),('absent',['--no-target']),('absent-time',['--no-target','--dt','15']),('absent-grid',['--no-target','--mesh-refinement','2']),('absent-domain',['--no-target','--set','Grid.UpperRight=2 2 .8','--set','Target.X=1','--set','Target.Y=1'])]
    if args.only:settings=[s for s in settings if s[0] in args.only]
    state=dict(status='running',stage=args.stage,virtual_duration_s=duration,started_at=time.time(),max_wall_seconds=args.max_wall_seconds,workers=args.workers,threads_per_worker=args.threads,research_accepted=False,jobs={name:dict(status='queued') for name,_ in settings})
    state.update(gpu_devices=devices,surface_refinement=args.surface_refinement,amgx_adapter=args.amgx_adapter)
    status=output/args.status_name;lock=threading.Lock();deadline=time.monotonic()+args.max_wall_seconds
    def save():
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
    save()
    def job(name,extra):
        case=output/name
        with lock:state['jobs'][name]=dict(status='running',started_at=time.time());save()
        command=[sys.executable,'-u',str(source/'run_dumux.py'),'--root',str(root/'upstream'),'--output-dir',str(case),'--duration',str(duration),'--dt','30','--threads',str(args.threads),'--cached','--ledger','--surface','--graded','--surface-refinement',str(args.surface_refinement),*(['--cached-parameters'] if args.cached_parameters else []),*(['--stream-surfaces'] if args.stream_surfaces else []),*(['--compact-output'] if args.compact_output else []),*(['--amgx'] if args.amgx_adapter else []),*extra]
        env=os.environ.copy();env['PYTHONPATH']=str(source.parent)
        device=tokens.get() if devices else None
        if device is not None:
            env['CUDA_VISIBLE_DEVICES']=str(device)
            command+=['--set','AMGX.Enabled=true','--set','AMGX.LibraryPath='+str(args.amgx_library)]
        code=124;started=time.monotonic()
        with (output/(name+'-driver.log')).open('w') as log:
            remaining=deadline-time.monotonic()
            if remaining>0:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        if device is not None:tokens.put(device)
        with lock:
            state['jobs'][name]=dict(status='completed' if code==0 else ('timed_out' if code==124 else 'failed'),exit_code=code,elapsed_wall_s=time.monotonic()-started,finished_at=time.time(),path=str(case));save()
        print(json.dumps(dict(job=name,**state['jobs'][name])),flush=True)
        return code
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes=[f.result() for f in as_completed([pool.submit(job,name,extra) for name,extra in settings])]
    state.update(status='completed' if all(c==0 for c in codes) else 'partial_or_failed',finished_at=time.time())
    save();print(json.dumps(state,indent=2));return 0 if all(c==0 for c in codes) else 1


if __name__=='__main__':raise SystemExit(main())
