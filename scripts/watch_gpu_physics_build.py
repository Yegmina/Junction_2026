"""Mirror actual CUDA build state into the research dashboard, without inference."""
import json
import shlex
import subprocess
import time

from aerorozum.common import data_root, redacted_error, save, utcnow
from aerorozum.jobs import job, update
from aerorozum.preflight import sdk_path

query = '''import json,pathlib,re,subprocess
root=pathlib.Path('/opt/aerorozum/gpu-physics')
log=(root/'build.log').read_text(errors='replace') if (root/'build.log').exists() else ''
percent=re.findall(r'\\[\\s*(\\d+)%\\]',log)
unit=subprocess.run(['systemctl','show','aerorozum-build-gpu-physics-r4.service','--property=ActiveState,SubState,Result'],capture_output=True,text=True).stdout
d={'unit':dict(line.split('=',1) for line in unit.splitlines() if '=' in line),'compiler_percent':int(percent[-1]) if percent else 0,'last_lines':log.splitlines()[-5:]}
if (root/'build-status.json').exists():d['last_exit_record']=json.loads((root/'build-status.json').read_text())
print(json.dumps(d))'''
record = job('gpu_physics_backend_build', completed=0, total=100,
             detail='Compile pinned NVIDIA AMGX; not yet an integrated physics result')
while True:
    try:
        response = subprocess.run(
            [sdk_path(), 'compute', 'ssh', 'aerorozum-gemma-pilot-20261114a',
             '--zone=europe-west4-a', '--project=avid-relic-509408-e4', '--quiet',
             '--ssh-flag=-oConnectTimeout=15',
             '--command=sudo /usr/bin/python3 -c ' + shlex.quote(query)],
            capture_output=True, text=True, timeout=45, check=True)
        state = json.loads(response.stdout)
        state['checked_at'] = utcnow()
        save(data_root()/'physics/gpu-build-progress.json', state)
        active = state['unit'].get('ActiveState')
        if active == 'failed':
            update(record, 'failed', error='CUDA build failed; original log retained', evidence=state)
            break
        if active == 'inactive':
            built = state.get('last_exit_record', {}).get('status') == 'built'
            update(record, 'completed' if built else 'failed', completed=100 if built else state['compiler_percent'], evidence=state)
            break
        update(record, completed=state['compiler_percent'], evidence=state)
    except Exception as exc:
        update(record, monitor_error=redacted_error(exc))
    time.sleep(30)
