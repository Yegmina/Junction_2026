"""Preserve an interrupted native trajectory, then rerun the identical full day.

The current DuMux binary has no restart-state reader. Surface/VTK samples must
never be presented as restart checkpoints or concatenated into a completed run.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', choices=['object-h4-local', 'object-h4-domain2'])
    a = p.parse_args()
    root = Path('/opt/aerorozum/physics')
    case = root/'cases/three-d-correspondence-v2'/a.case
    state_path = case/'run-status.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status') != 'completed':
        with (case/'.run.lock').open('a') as lock:
            # Refuse to move a directory belonging to an active native runner.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archived = root/'interrupted'/stamp/a.case
            archived.parent.mkdir(parents=True, exist_ok=True)
            case.rename(archived)
            original = (archived/'run-status.json').read_bytes()
            (archived/'original-run-status.json').write_bytes(original)
            prior = json.loads(original)
            prior.update(status='interrupted_preserved', research_accepted=False,
                         interrupted_at=datetime.now(timezone.utc).isoformat())
            (archived/'run-status.json').write_text(json.dumps(prior, indent=2))
            log = case.parent/(a.case+'-driver.log')
            if log.exists():
                shutil.copy2(log, archived/'driver-before-restart.log')
            (archived/'retirement.json').write_text(json.dumps({
                'status': 'interrupted_preserved', 'research_accepted': False,
                'reason': 'VM/service restart; no solver restart-state reader exists. Rerun identical full day in fresh directory.',
                'recorded_at': datetime.now(timezone.utc).isoformat()}, indent=2))
    command = ['/opt/aerorozum/venv/bin/python', str(root/'source/physics/run_dumux.py'),
        '--root', str(root/'upstream'), '--output-dir', str(case), '--duration', '86400',
        '--dt', '30', '--threads', '16', '--cached', '--ledger', '--surface', '--graded',
        '--cached-parameters', '--compact-output', '--amgx', '--surface-refinement', '2',
        '--horizontal-refinement', '4', '--vertical-refinement', '2']
    if a.case.endswith('domain2'):
        command += ['--set', 'Grid.UpperRight=2 2 .4', '--set', 'Target.X=1', '--set', 'Target.Y=1']
    env = dict(os.environ, PYTHONPATH=str(root/'source'))
    with (case.parent/(a.case+'-driver.log')).open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
