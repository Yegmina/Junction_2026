#!/usr/bin/env python3
"""Summarize completed real pilot comparisons; never admit a short campaign."""
import argparse
import json
from pathlib import Path

from verify_dumux_cases import compare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.root;checks={}
    for group,base,names in [('present','present',{'time':'time','grid':'grid','domain':'domain','regularization':'regularization'}),('absent','absent',{'time':'absent-time','grid':'absent-grid','domain':'absent-domain'})]:
        for kind,directory in names.items():checks[group+'-'+kind]=compare(root/base/'field.npz',root/directory/'field.npz',coordinate_shift_m=.5 if kind=='domain' else 0.)
    state=json.loads((root/'suite-status.json').read_text())
    result=dict(status='pilot_passed' if all(c['status']=='passed' for c in checks.values()) else 'pilot_pending_or_failed',virtual_duration_s=state['virtual_duration_s'],research_accepted=False,checks=checks,jobs=state['jobs'],next_gate='Full-day convergence and per-trajectory actual DuMux budgets still required')
    output=a.output or root/'pilot-screening.json';output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['status']=='pilot_passed' else 1


if __name__=='__main__':raise SystemExit(main())
