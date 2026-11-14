#!/usr/bin/env python3
"""Install safe-point gates in an old worker without cancelling paid requests."""
import argparse,json,subprocess,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('pid',type=int);p.add_argument('--receipt',required=True);a=p.parse_args();receipt=Path(a.receipt).resolve()
source=(Path(__file__).with_name('legacy_drain_payload.py')).read_text().replace('RECEIPT_PATH',repr(str(receipt)))
with tempfile.NamedTemporaryFile('w',prefix='aerorozum-drain-',suffix='.py',delete=False) as f:f.write(source);payload=f.name
code='exec(compile(open('+repr(payload)+').read(),'+repr(payload)+',"exec"))'
commands='set pagination off\nset confirm off\nset $aero_buf=(char*)malloc('+str(len(code.encode())+1)+')\nset {char['+str(len(code.encode())+1)+']} $aero_buf='+json.dumps(code)+'\ncall (int)Py_AddPendingCall((int (*)(void *))PyRun_SimpleString,(void*)$aero_buf)\ndetach\nquit\n'
with tempfile.NamedTemporaryFile('w',suffix='.gdb',delete=False) as f:f.write(commands);command=f.name
try:
 subprocess.run(['sudo','-n','gdb','-q','-nx','-batch','-p',str(a.pid),'-x',command],check=True,timeout=25)
 for _ in range(100):
  if receipt.exists():print(receipt.read_text());Path(payload).unlink(missing_ok=True);break
  time.sleep(.1)
 else:raise RuntimeError('No drain receipt yet; payload retained and worker not killed')
finally:Path(command).unlink(missing_ok=True)
