#!/usr/bin/env python3
"""Check an execution optimization against actual identical-scenario fields."""
import argparse
import json
from pathlib import Path

from verify_dumux_cases import compare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('optimized',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.reference/'field.npz',a.optimized/'field.npz',tolerance_k=.0001)
    before=json.loads((a.reference/'run-status.json').read_text());after=json.loads((a.optimized/'run-status.json').read_text())
    result.update(reference_wall_s=before['elapsed_wall_s'],optimized_wall_s=after['elapsed_wall_s'],observed_speed_ratio=before['elapsed_wall_s']/after['elapsed_wall_s'],timing_context='Same VM/scenario, concurrent CPU workloads; not an isolated scaling benchmark',research_accepted=False)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
