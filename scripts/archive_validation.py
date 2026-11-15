#!/usr/bin/env python3
"""Continuously archive validation evidence and original training journal to GCS."""
import argparse,hashlib,json,time,urllib.request,urllib.parse
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bucket',required=True);p.add_argument('--prefix',required=True);p.add_argument('--journal',required=True);p.add_argument('--seconds',type=int,default=60);a=p.parse_args()
 root=Path(a.root);uploaded={};token=None;token_until=0
 def upload(path,name):
  nonlocal token,token_until
  body=path.read_bytes();sha=hashlib.sha256(body).hexdigest()
  if uploaded.get(name)==sha:return
  if time.monotonic()>token_until:
   request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers={'Metadata-Flavor':'Google'})
   d=json.load(urllib.request.urlopen(request,timeout=15));token=d['access_token'];token_until=time.monotonic()+max(30,d['expires_in']-120)
  url='https://storage.googleapis.com/upload/storage/v1/b/'+a.bucket+'/o?uploadType=media&name='+urllib.parse.quote(name,safe='')
  request=urllib.request.Request(url,data=body,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json' if path.suffix=='.json' else 'text/plain'})
  with urllib.request.urlopen(request,timeout=120) as response:response.read()
  uploaded[name]=sha
 while True:
  try:
   for path in sorted(root.rglob('*')):
    if path.is_file() and not path.name.endswith('.tmp'):upload(path,a.prefix.rstrip('/')+'/validation/'+path.relative_to(root).as_posix())
   journal=Path(a.journal)
   if journal.is_file():upload(journal,a.prefix.rstrip('/')+'/logs/'+journal.name)
   print(json.dumps({'uploaded_objects':len(uploaded),'at':time.time()}),flush=True)
   status=root/'status.json'
   if status.exists() and json.loads(status.read_text()).get('status') in ('completed','failed'):break
   if (root/'failure.json').exists():break
  except Exception as ex:print(json.dumps({'archive_error':str(ex)[:500]}),flush=True)
  time.sleep(a.seconds)
if __name__=='__main__':main()
