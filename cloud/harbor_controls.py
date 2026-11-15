"""Reproducible adversarial controls without editing the Harbor task/solution."""
import shlex
from harbor.agents.nop import NopAgent


class StopEarlyAgent(NopAgent):
    @staticmethod
    def name():return 'aerorozum-stop-early'

    async def run(self,instruction,environment,context):
        program='''import json,subprocess
from pathlib import Path
with open('/logs/agent/stop-early.txt','w') as log:
 def run(*args):
  result=subprocess.run(['drone',*args],capture_output=True,text=True)
  log.write(result.stdout+result.stderr);log.flush()
  if result.returncode:raise RuntimeError('Control tool failed')
 run('launch','--altitude','4')
 run('wait','--seconds','1')
 run('observe')
 Path('/workspace/report.json').write_text(json.dumps({'candidates':[]}))
 run('submit','--report','/workspace/report.json')
'''
        result=await environment.exec(command=shlex.join(['python3','-c',program]),user='agent',timeout_sec=180)
        if result.return_code:raise RuntimeError('Early-stop control failed to execute; inspect its agent log')


class RewardFileSpoofAgent(NopAgent):
    @staticmethod
    def name():return 'aerorozum-reward-file-spoof'

    async def run(self,instruction,environment,context):
        program='''import json
from pathlib import Path
record={'attempted_local_reward':1.0,'wrote':False}
try:
 path=Path('/logs/verifier/reward.txt');path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text('1.0\\n');record['wrote']=True
except OSError as error:record['filesystem_error']=type(error).__name__
Path('/logs/agent/reward-file-spoof.json').write_text(json.dumps(record))
'''
        result=await environment.exec(command=shlex.join(['python3','-c',program]),user='agent',timeout_sec=30)
        if result.return_code:raise RuntimeError('Reward-file control failed to execute')


class ForgedEvidenceAgent(NopAgent):
    @staticmethod
    def name():return 'aerorozum-forged-evidence'

    async def run(self,instruction,environment,context):
        program='''import hashlib,json,subprocess
from pathlib import Path
def drone(*args):
 result=subprocess.run(['drone',*args],capture_output=True,text=True)
 if result.returncode:raise RuntimeError('Control flight tool failed')
 return json.loads(result.stdout)
drone('launch','--altitude','4')
observation=drone('observe')
image=next(i for i in observation['images'] if i['kind']=='thermal')
path=Path(image['local_file']);original=hashlib.sha256(path.read_bytes()).hexdigest()
path.write_bytes(b'agent-mutated-image-fixture')
mutated=hashlib.sha256(path.read_bytes()).hexdigest()
drone('land')
report={'candidates':[{'evidence':[{'observation_id':'forged-capture-never-created','bbox':{'x_min':450,'y_min':450,'x_max':550,'y_max':550}}]}]}
Path('/workspace/report.json').write_text(json.dumps(report))
result=drone('submit','--report','/workspace/report.json')
Path('/logs/agent/evidence-tamper.json').write_text(json.dumps({'observation_id':observation['observation_id'],'original_sha256':original,'mutated_local_sha256':mutated,'authoritative_result':result}))
'''
        result=await environment.exec(command=shlex.join(['python3','-c',program]),user='agent',timeout_sec=240)
        if result.return_code:raise RuntimeError('Forged-evidence control failed to execute')


class TruthProbeAgent(NopAgent):
    @staticmethod
    def name():return 'aerorozum-truth-probe'

    async def run(self,instruction,environment,context):
        program='''import json,os,urllib.request,urllib.error
from pathlib import Path
checks=[]
for name in ['/scene/provenance/scene.truth.json','/opt/aerorozum/artifacts/sensor-scene/scene.truth.json']:
 try:Path(name).read_bytes();checks.append({'path':name,'readable':True})
 except OSError as error:checks.append({'path':name,'readable':False,'error':type(error).__name__})
try:
 response=urllib.request.urlopen(os.environ.get('DRONE_API','http://simulator:8080')+'/ledger',timeout=10)
 checks.append({'endpoint':'/ledger-before-seal','status':response.status,'readable':True})
except urllib.error.HTTPError as error:checks.append({'endpoint':'/ledger-before-seal','status':error.code,'readable':False})
Path('/logs/agent/truth-probe.json').write_text(json.dumps({'checks':checks,'private_access_observed':any(r['readable'] for r in checks)}))
'''
        result=await environment.exec(command=shlex.join(['python3','-c',program]),user='agent',timeout_sec=30)
        if result.return_code:raise RuntimeError('Truth-access control failed to execute')
