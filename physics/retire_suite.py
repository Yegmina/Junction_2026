#!/usr/bin/env python3
"""Record an already-stopped suite as superseded, retaining all evidence."""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--reason',required=True)
a=p.parse_args();status=a.root/'suite-status.json';data=json.loads(status.read_text())
for name,job in data['jobs'].items():
    if job['status'] in ('running','queued'):
        job.update(status='superseded',reason=a.reason,stopped_at=time.time())
        path=a.root/name/'run-status.json'
        if path.exists():
            prior=json.loads(path.read_text());prior.update(status='superseded',reason=a.reason,stopped_at=time.time());path.write_text(json.dumps(prior,indent=2))
data.update(status='superseded',reason=a.reason,stopped_at=time.time());status.write_text(json.dumps(data,indent=2))
print(json.dumps(dict(status='superseded',root=str(a.root),reason=a.reason)))
