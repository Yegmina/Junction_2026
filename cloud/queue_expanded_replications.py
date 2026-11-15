"""Admit expanded seeds only after the original fixed-recipe run releases its GPU."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


if __name__=='__main__':
    root=Path('/opt/aerorozum');pending={43:2,91:3}
    deadline=datetime.fromisoformat('2026-11-15T04:00:00+00:00')
    while pending and datetime.now(timezone.utc)<deadline:
        for seed,gpu in list(pending.items()):
            parent=root/f'artifacts/postsubmission-visual-replication-v1/seed-{seed}/status.json'
            if not parent.exists():continue
            state=json.loads(parent.read_text())
            if state.get('status') not in ('completed','failed'):continue
            if (root/f'artifacts/postsubmission-expanded-visual-v1/seed-{seed}').exists():
                del pending[seed];continue
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('launch_expanded_visual.py')),
                '--seed',str(seed),'--gpu',str(gpu),'--optimized-monitor'],capture_output=True,text=True,timeout=300)
            print(json.dumps(dict(seed=seed,gpu=gpu,parent_status=state['status'],exit_code=result.returncode,
                output=result.stdout[-2000:],error=result.stderr[-1200:])),flush=True)
            if result.returncode==0:del pending[seed]
            else:
                # Preserve the first rejected admission for diagnosis; do not
                # repeatedly create partial experiment folders.
                Path(root/f'artifacts/overnight-20261115T000942/expanded-admission-error-{seed}.json').write_text(json.dumps(dict(stdout=result.stdout[-3000:],stderr=result.stderr[-3000:])))
                del pending[seed]
        time.sleep(20)
