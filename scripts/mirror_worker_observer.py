"""Mirror public worker state and previews through a private SSH port forward."""
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

from aerorozum.common import data_root, save, utcnow, redacted_error

root = data_root()
base = 'http://127.0.0.1:18920'
media_cache = {}
while True:
    try:
        with urllib.request.urlopen(base+'/snapshot', timeout=10) as response:
            state = json.load(response)
        state['mirrored_at'] = utcnow()
        for section in ('physics', 'controller'):
            document = state.get(section) or {}
            media = document.get('media', [])
            stamp = document.get('updated_at') or document.get('checked_at')
            for row in media:
                relative = row.get('image', '')
                path = Path(relative)
                if not relative.startswith('pipeline/media/') or '..' in path.parts:
                    continue
                target = root/path
                prior = media_cache.get(relative, {})
                if stamp and prior.get('stamp')==stamp and target.is_file():
                    row['sha256'] = prior['sha256']
                    continue
                suffix = relative.removeprefix('pipeline/media/')
                try:
                    with urllib.request.urlopen(base+'/media/'+urllib.parse.quote(suffix, safe='/'), timeout=10) as response:
                        payload = response.read(16*1024**2)
                except (OSError, TimeoutError):
                    row['media_unavailable'] = True
                    continue
                sha = hashlib.sha256(payload).hexdigest()
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(target.suffix+'.tmp');temp.write_bytes(payload);temp.replace(target)
                row['sha256'] = sha
                media_cache[relative] = {'stamp': stamp, 'sha256': sha}
        save(root/'pipeline/worker.json', state)
        save(root/'pipeline/telemetry.json', state.get('telemetry', {}))
        save(root/'training/status.json', state.get('training', {}))
        save(root/'pipeline/connection.json', {'status':'connected', 'checked_at': utcnow()})
    except Exception as exc:
        save(root/'pipeline/connection.json', {'status':'reconnecting', 'checked_at':utcnow(), 'error':redacted_error(exc)})
    time.sleep(3)
