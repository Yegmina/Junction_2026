"""Refresh and archive scientific native-outcome snapshots as real runs finish."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/opt/aerorozum');NIGHT=ROOT/'artifacts/overnight-20261115T000942'
PUBLIC=ROOT/'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'
END=datetime.fromisoformat('2026-11-15T06:35:00+00:00')


def main():
    from aerorozum.common import save,utcnow
    previous=None
    while datetime.now(timezone.utc)<END:
        terminal=[]
        for path in (ROOT/'harbor/jobs').glob('overnight-*/result.json'):
            try:
                data=path.read_bytes();value=json.loads(data);stats=value.get('stats',{})
                if value.get('finished_at') and not stats.get('n_pending_trials') and not stats.get('n_running_trials'):
                    terminal.append((str(path),hashlib.sha256(data).hexdigest()))
            except (OSError,ValueError):continue
        signature=hashlib.sha256(json.dumps(sorted(terminal)).encode()).hexdigest()
        if signature!=previous:
            try:
                subprocess.run([sys.executable,str(Path(__file__).with_name('export_native_investigations.py'))],check=True,timeout=120)
                out=NIGHT/'native-outcomes-v1'
                for name in ['native-outcomes-v1.png','native-outcomes-v1.pdf','summary.json']:
                    target=PUBLIC/(name if name!='summary.json' else 'native-outcomes-summary.json');temporary=target.with_name('.'+target.name+'.next');shutil.copyfile(out/name,temporary);temporary.chmod(0o644);temporary.replace(target)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                subprocess.run(['/snap/bin/gcloud','storage','cp',str(out/'native-outcomes-v1.png'),str(out/'native-outcomes-v1.pdf'),str(out/'summary.json'),
                    'gs://aerorozum-thermal-rl-avid-relic-509408-e4/research/overnight-20261115/native-outcome-snapshots/'+stamp+'/','--quiet'],check=True,timeout=300)
                save(NIGHT/'native-visuals-status.json',dict(at=utcnow(),terminal_count=len(terminal),signature=signature,cloud_snapshot=stamp))
                previous=signature
            except Exception as error:print(type(error).__name__+': '+str(error)[:1000],flush=True)
        time.sleep(60)


if __name__=='__main__':main()
