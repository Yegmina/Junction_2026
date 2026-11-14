"""Read-only worker telemetry over loopback; reachable only through the SSH tunnel."""
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit

ROOT = Path('/opt/aerorozum/artifacts')
PIPELINE = ROOT/'pipeline'
MEDIA = PIPELINE/'media'
LOCK = threading.Lock()
STATE = {'telemetry': {}, 'history': []}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def cpu():
    values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return sum(values[:8]), values[3]+values[4]


def sample():
    previous = cpu()
    history = deque(maxlen=300)
    PIPELINE.mkdir(parents=True, exist_ok=True)
    while True:
        time.sleep(2)
        now = cpu()
        elapsed = now[0]-previous[0]
        used = 100*(elapsed-(now[1]-previous[1]))/max(1, elapsed)
        previous = now
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.split()[0])
        disk = shutil.disk_usage(ROOT)
        row = {'checked_at': datetime.now(timezone.utc).isoformat(),
               'time': time.time(), 'cpu_percent': used, 'cpu_cores': 48,
               'ram_percent': 100*(1-memory['MemAvailable']/memory['MemTotal']),
               'ram_used_gib': (memory['MemTotal']-memory['MemAvailable'])/1024**2,
               'ram_total_gib': memory['MemTotal']/1024**2,
               'disk_free_gib': disk.free/1024**3, 'gpus': [],
               'utilization_target_percent': 90, 'cpu_core_limit': 43}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
            for line in result.stdout.splitlines():
                index, utilization, used_memory, total_memory = [float(x.strip()) for x in line.split(',')]
                row['gpus'].append({'index': int(index), 'utilization_percent': utilization,
                                    'memory_used_mib': used_memory, 'memory_total_mib': total_memory})
        except (ValueError, OSError, subprocess.SubprocessError):
            row['gpu_read_error'] = True
        row['gpu_percent'] = sum(g['utilization_percent'] for g in row['gpus'])/len(row['gpus']) if row['gpus'] else None
        history.append({k: row[k] for k in ('time', 'cpu_percent', 'gpu_percent', 'ram_percent')})
        with LOCK:
            STATE.update(telemetry=row, history=list(history))
        temporary = PIPELINE/'telemetry.json.tmp'
        temporary.write_text(json.dumps(row)); temporary.replace(PIPELINE/'telemetry.json')
        with (PIPELINE/'telemetry-history.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route == '/snapshot':
            with LOCK:
                document = dict(STATE)
            document.update(physics=read(PIPELINE/'physics.json', {}),
                            controller=read(PIPELINE/'status.json', {}),
                            labelling=read(Path('/opt/aerorozum/airborne-labeler/artifacts/pipeline/label-worker.json'), {}),
                            training=read(ROOT/'training/status.json', {}))
            content = json.dumps(document).encode()
            mime = 'application/json'
        elif route.startswith('/media/'):
            target = (MEDIA/route.removeprefix('/media/')).resolve()
            if not target.is_relative_to(MEDIA.resolve()) or target.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp') or not target.is_file():
                self.send_error(404); return
            content = target.read_bytes()
            mime = 'image/png' if target.suffix.lower()=='.png' else 'image/jpeg'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)


if __name__ == '__main__':
    threading.Thread(target=sample, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', 18920), Handler).serve_forever()
