"""Restore boot persistence for the existing, preserved research services."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess


def main():
    root=Path('/opt/aerorozum/deployment')
    backup=root/'before-48h'
    names=[]
    # The backup contains only services that were active before the move,
    # plus the two explicitly preserved interrupted native cases.
    for path in backup.glob('aerorozum-*.service'):
        if path.name.startswith(('aerorozum-cloud-','aerorozum-deploy-')):
            continue
        target=Path('/etc/systemd/system')/path.name
        if not target.is_file():
            raise RuntimeError('Preserved service definition missing: '+path.name)
        text=target.read_text()
        if 'upload' in path.name or 'admission' in path.name or 'live-progress' in path.name:
            text=re.sub(r'^RuntimeMaxSec=.*$', 'RuntimeMaxSec=169200', text, flags=re.M)
        if path.name=='aerorozum-physics-controller.service':
            deadline=(datetime.now(timezone.utc)+timedelta(hours=47)).isoformat()
            text=re.sub(r'--deadline \S+', '--deadline '+deadline, text)
            text=re.sub(r'^RuntimeMaxSec=.*$', 'RuntimeMaxSec=169200', text, flags=re.M)
        target.write_text(text)
        names.append(path.name)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    # add-wants also works for the original static uploader units.
    subprocess.run(['systemctl','add-wants','multi-user.target',*names],check=True)
    subprocess.run(['systemctl','start',*names],check=True)
    (root/'recovery-receipt.json').write_text(json.dumps(dict(
        at=datetime.now(timezone.utc).isoformat(),services=names,
        interrupted_physics='Original partial runs archived; identical full-day reruns',
        backup=str(backup)),indent=2))
    print(json.dumps({'boot_persistent_research_services':names}))


if __name__=='__main__':
    main()
