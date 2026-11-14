#!/usr/bin/env python3
"""Install on the existing VM without starting paid inference."""
import argparse,json,os,pwd,re,shutil,subprocess,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--key-basename',required=True);a=p.parse_args()
    if os.geteuid()!=0:raise ValueError('Existing VM installation requires root')
    if not re.fullmatch(r'aerorozum-label-key-[A-Za-z0-9_-]+',a.key_basename):raise ValueError('Unexpected credential filename')
    home=Path('/opt/aerorozum/airborne-labeler');source=home/'source'
    try:account=pwd.getpwnam('aerorozum-labeler')
    except KeyError:
        subprocess.run(['useradd','--system','--home-dir',str(home/'state'),'--no-create-home','--shell','/usr/sbin/nologin','aerorozum-labeler'],check=True);account=pwd.getpwnam('aerorozum-labeler')
    source.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.bundle) as archive:
        for member in archive.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or member.issym() or member.islnk():raise ValueError('Unsafe archive member')
        archive.extractall(source,filter='data')
    for path in [source,*source.rglob('*')]:
        os.chown(path,0,0);path.chmod(0o755 if path.is_dir() else 0o644)
    for name in ['artifacts','state']:
        path=home/name;path.mkdir(exist_ok=True);os.chown(path,account.pw_uid,account.pw_gid);path.chmod(0o750)
    private=Path('/etc/aerorozum-labeler');private.mkdir(mode=0o700,exist_ok=True);private.chmod(0o700)
    temporary=Path('/tmp')/a.key_basename
    target=private/'gemini-api-key'
    if temporary.exists():
        if temporary.stat().st_mode&0o077:raise ValueError('Transferred credential is not private')
        shutil.copyfile(temporary,target);target.chmod(0o600);os.chown(target,0,0);temporary.unlink()
    elif not target.is_file() or target.stat().st_mode&0o077:raise ValueError('Installed private credential unavailable')
    unit=Path('/etc/systemd/system/aerorozum-airborne-labeler.service');shutil.copy2(source/'cloud/aerorozum-airborne-labeler.service',unit);unit.chmod(0o644)
    subprocess.run(['systemctl','daemon-reload'],check=True);subprocess.run(['systemctl','enable',unit.name],check=True)
    subprocess.run(['systemd-analyze','verify',str(unit)],check=True)
    print(json.dumps({'status':'installed_inactive_no_paid_calls','service':unit.name,'credential_mode':oct(target.stat().st_mode&0o777),'launch_authorization_present':(home/'state/launch.json').exists()}))

if __name__=='__main__':main()
