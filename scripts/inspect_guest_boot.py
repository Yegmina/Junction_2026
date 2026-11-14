#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
from aerorozum.common import data_root,save
from aerorozum.preflight import sdk_path
remote="sudo python3 - <<'PYREMOTE'\n"+(Path(__file__).parent/'guest_boot_probe.py').read_text()+"\nPYREMOTE"
r=subprocess.run([sdk_path(),'compute','ssh','aerorozum-gemma-pilot-20261114a','--zone','europe-west4-a','--project','avid-relic-509408-e4','--command',remote,'--quiet'],capture_output=True,text=True,timeout=60)
if r.returncode:raise RuntimeError(r.stderr[-1000:])
d=json.loads(r.stdout);save(data_root()/'cloud/guest-boot-diagnostics-20261114.json',d);print(json.dumps({'at':d['collected_at'],'apt_nov14_transactions':len(d['apt_nov14']),'pcr7_events':len(d.get('pcr7_events',[])),'pcr7_replayed':d.get('pcr7_sha256_replayed'),'tpm_parse_error':d.get('tpm_parse_error')},indent=2))
