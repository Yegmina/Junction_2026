#!/usr/bin/env python3
"""Record explicitly stopped, superseded reference cases without deleting data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--reason',required=True)
a=p.parse_args();now=time.time();audit=[]
for name in a.cases:
    case=a.root/name;path=case/'run-status.json'
    if not path.exists():continue
    state=json.loads(path.read_text())
    if state['status']=='completed':continue
    state.update(status='retired_superseded',retired_at=now,retirement_reason=a.reason,research_accepted=False)
    path.write_text(json.dumps(state,indent=2));audit.append(name)
    for file in a.root.glob('*status.json'):
        suite=json.loads(file.read_text())
        if name in suite.get('jobs',{}):
            suite['jobs'][name].update(status='retired_superseded',reason=a.reason,retired_at=now)
            file.write_text(json.dumps(suite,indent=2))
proof=dict(status='recorded_3d_grid_failure',research_accepted=False,reason=a.reason,retired_cases=audit,recorded_at=now,evidence={})
pipeline=Path('/opt/aerorozum/artifacts/pipeline/physics.json')
if pipeline.exists():proof['latest_physics_snapshot']=json.loads(pipeline.read_text())
for name in ['present','grid']:
    case=a.root/name;entry={}
    for filename in ['run-status.json','surface-frames.bin','surface-grid.csv','surface-format.json','conservation.csv']:
        path=case/filename
        if path.exists():
            size=path.stat().st_size
            with path.open('rb') as source:data=source.read(size)
            entry[filename]=dict(prefix_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    proof['evidence'][name]=entry
(a.root/'recorded-3d-grid-failure.json').write_text(json.dumps(proof,indent=2));print(json.dumps(dict(retired_cases=audit,proof=str(a.root/'recorded-3d-grid-failure.json'))))
