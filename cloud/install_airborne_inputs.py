#!/usr/bin/env python3
"""Fetch generation-pinned frozen pixels on the existing VM; never infer."""
import argparse,hashlib,json,os,pwd,tarfile
from pathlib import Path
from urllib.parse import quote,urlsplit
from google.auth.compute_engine import Credentials
from google.auth.transport.requests import AuthorizedSession

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);a=p.parse_args();d=json.loads(a.receipt.read_text());home=Path('/opt/aerorozum/airborne-labeler');root=home/'artifacts';archive=home/'state/frozen-inputs.tar.gz'
    parsed=urlsplit(d['uri'])
    if parsed.scheme!='gs' or parsed.netloc!='aerorozum-thermal-rl-avid-relic-509408-e4':raise ValueError('Reviewed run bucket required')
    credentials=Credentials(service_account_email='94824551802-compute@developer.gserviceaccount.com',scopes=['https://www.googleapis.com/auth/cloud-platform']);session=AuthorizedSession(credentials)
    response=session.get('https://storage.googleapis.com/storage/v1/b/'+parsed.netloc+'/o/'+quote(parsed.path.lstrip('/'),safe=''),params={'alt':'media','generation':d['generation']},stream=True,timeout=180);response.raise_for_status();h=hashlib.sha256()
    with archive.open('wb') as out:
        for part in response.iter_content(1024*1024):out.write(part);h.update(part)
    if h.hexdigest()!=d['sha256'] or archive.stat().st_size!=d['bytes']:raise ValueError('Frozen input archive checksum differs')
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or member.issym() or member.islnk():raise ValueError('Unsafe input archive member')
        source.extractall(root,filter='data')
    account=pwd.getpwnam('aerorozum-labeler')
    for path in [root,*root.rglob('*')]:
        os.chown(path,account.pw_uid,account.pw_gid);path.chmod(0o750 if path.is_dir() else 0o640)
    (home/'state/input-staging.json').write_text(json.dumps(d,indent=2));os.chown(home/'state/input-staging.json',account.pw_uid,account.pw_gid)
    print(json.dumps({'status':'installed_exact_frozen_inputs','rows':d['rows'],'files':d['files'],'manifest':str(root/d['manifest']),'paid_calls':0}))

if __name__=='__main__':main()
