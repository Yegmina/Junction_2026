"""Mirror measured training/job status to the public ledger without a UI restart."""
import json,time,shutil
from pathlib import Path
source=Path('/opt/aerorozum/artifacts');target=Path('/opt/aerorozum/deployment/artifacts')
while True:
 for path in (source/'jobs').glob('*.json'):
  try:
   destination=target/'jobs'/path.name
   if not destination.exists() or json.loads(path.read_text()).get('updated_at','')>json.loads(destination.read_text()).get('updated_at',''):
    temp=destination.with_suffix('.mirror-tmp');shutil.copyfile(path,temp);temp.replace(destination)
  except (OSError,ValueError):pass
 try:
  snapshot=target/'status.json';document=json.loads(snapshot.read_text())
  jobs=[json.loads(p.read_text()) for p in (target/'jobs').glob('*.json')]
  document['jobs']=sorted(jobs,key=lambda job:job.get('updated_at',job.get('created_at','')),reverse=True)
  temp=snapshot.with_suffix('.ledger-tmp');temp.write_text(json.dumps(document));temp.replace(snapshot)
 except (OSError,ValueError):pass
 time.sleep(10)
