#!/usr/bin/env python3
"""Keep the existing CLI credential cache fresh; never print token values."""
import datetime,json,os,subprocess,sys,tempfile,time
from pathlib import Path
sdk=Path('/home/yehormsi/.local/google-cloud-cli-install/google-cloud-sdk');sys.path[:0]=[str(sdk/'lib'),str(sdk/'lib/third_party')]
from googlecloudsdk.core.credentials import store
output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True)
def save(record):
 fd,name=tempfile.mkstemp(dir=output.parent,prefix='.sdk-token-cache-',suffix='.tmp')
 try:
  with os.fdopen(fd,'w') as f:json.dump(record,f,indent=2);f.write('\n')
  os.replace(name,output)
 finally:Path(name).unlink(missing_ok=True)
while True:
 try:
  before=store.Load();previous=before.expiry;fresh=store.LoadFreshCredential(min_expiry_duration='55m');expiry=fresh.expiry.replace(tzinfo=datetime.timezone.utc) if fresh.expiry.tzinfo is None else fresh.expiry
  cli=subprocess.check_output([str(sdk/'bin/gcloud'),'auth','print-access-token','--quiet'],text=True,timeout=30).strip();matching=cli==fresh.token
  if not matching:raise RuntimeError('CLI cache refresh verification failed')
  record={'status':'active','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expiry_utc':expiry.isoformat(),'remaining_seconds':(expiry-datetime.datetime.now(datetime.timezone.utc)).total_seconds(),'refreshed':previous!=fresh.expiry,'cli_cache_matches':matching,'minimum_refresh_window':'55m','interval_seconds':120,'token_values_exposed':False,'main_process_modified':False};save(record);print(json.dumps(record),flush=True)
 except Exception as ex:
  record={'status':'error','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error_type':type(ex).__name__,'token_values_exposed':False};save(record);print(json.dumps(record),flush=True)
 time.sleep(120)
