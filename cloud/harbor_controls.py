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
