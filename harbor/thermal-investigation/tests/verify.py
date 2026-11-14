"""Grade the sidecar's immutable result, never an agent-supplied reward file."""
import hashlib,json,math,urllib.request
from pathlib import Path
output=Path('/logs/verifier');output.mkdir(parents=True,exist_ok=True)
reward=0.
try:
 request=urllib.request.Request('http://simulator:8080/seal',data=b'{}',headers={'Content-Type':'application/json'})
 result=json.load(urllib.request.urlopen(request,timeout=120))
 reward=result['reward']
 if isinstance(reward,bool) or not isinstance(reward,(float,int)) or not math.isfinite(reward) or not 0<=reward<=1:raise ValueError('Invalid verifier result')
 (output/'details.json').write_text(json.dumps(result,indent=2))
 ledger=json.load(urllib.request.urlopen('http://simulator:8080/ledger',timeout=30))
 (output/'ledger.json').write_text(json.dumps(ledger,indent=2))
 evidence=output/'evidence';evidence.mkdir(exist_ok=True)
 for row in ledger.get('observations',{}).values():
  for item in row['images']:
   payload=urllib.request.urlopen('http://simulator:8080'+item['url'],timeout=30).read(3000000)
   if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('Archived image hash mismatch')
   (evidence/Path(item['filename']).name).write_bytes(payload)
except Exception as error:
 reward=0.;(output/'error.txt').write_text(str(error))
(output/'reward.txt').write_text(f'{reward:.8f}\n')
