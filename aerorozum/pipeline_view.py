"""Read-only presentation of measured pipeline artifacts; never starts work."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

_INPUT_CACHE = {}
_INPUT_LOCK = threading.Lock()


def input_summary(path):
    """A dashboard must not send thousands of training records every three seconds."""
    try:
        stat=path.stat();signature=(stat.st_mtime_ns,stat.st_size)
    except OSError:
        return {}
    with _INPUT_LOCK:
        cached=_INPUT_CACHE.get(str(path))
        if cached and cached[0]==signature:
            return cached[1]
        document=read(path,{}) or {}
        keep={'schema_version','updated_at','counts','cache_counts','cloud_sync','failure_reasons',
              'background_manifest','quarantined_sources','recovery','limitations','dataset_signature'}
        summary={k:v for k,v in document.items() if k in keep}
        _INPUT_CACHE[str(path)]=(signature,summary)
        return summary


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def snapshot(root):
    root = Path(root)
    worker = read(root/'pipeline/worker.json', {}) or {}
    telemetry = worker.get('telemetry') or read(root/'pipeline/telemetry.json', {})
    stamp = telemetry.get('checked_at')
    try:
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(stamp.replace('Z','+00:00'))).total_seconds()
    except (TypeError, ValueError, AttributeError):
        age = None
    labels = read(root/'labelling/robotics-er2-5fps/labels/progress.json', {})
    label_worker = worker.get('labelling') or read(root/'pipeline/label-worker-transition.json', {})
    if label_worker:
        labels = {**labels, **label_worker.get('label_progress', {}),
                  'status':label_worker.get('status', labels.get('status')),
                  'worker_host':label_worker.get('host'),
                  'label_prompt_version':label_worker.get('prompt_version'),
                  'label_prompt_sha256':label_worker.get('prompt_sha256')}
    sampling = read(root/'labelling/robotics-er2-5fps/sampling-progress.json', {})
    inventory = read(root/'labelling/eta-measurements.json', {})
    inputs = input_summary(root/'pipeline/inputs/real.json')
    controller = worker.get('controller') or read(root/'pipeline/status.json', {})
    physics = worker.get('physics') or read(root/'pipeline/physics.json', {})
    media = []
    for source, document in (('physics', physics), ('synthetic', controller)):
        for row in document.get('media', []):
            relative = row.get('image', '')
            target = (root/relative).resolve()
            if relative.startswith('pipeline/media/') and target.is_relative_to((root/'pipeline/media').resolve()) and target.is_file():
                kind=source if source=='physics' else row.get('kind') or ('synthetic' if row.get('provenance') in {'direct_physics','synthetic_verified','validated_surrogate'} else 'training')
                media.append({**row, 'kind':kind})
    return {'updated_at': datetime.now(timezone.utc).isoformat(),
            'worker_connected': age is not None and age < 30,
            'telemetry_age_s': age, 'telemetry':telemetry, 'history':worker.get('history', []),
            'policy': {'utilization_target_percent':90, 'cpu_core_limit':43,
                       'ram_limit_gib':300, 'gpu_memory_fraction_limit':.90},
            'controller':controller, 'physics':physics, 'inputs':inputs,
            'labels':labels, 'sampling':sampling,
            'label_worker':label_worker,
            'prompt_history':read(root/'labelling/prompt-history/summary.json', {}),
            'estimated_source_frames':inventory.get('estimated_5fps_frames'),
            'training':read(root/'training/status.json', {}),
            'connection':read(root/'pipeline/connection.json', {}),
            'cloud_health':read(root/'cloud/health.json', {}),
            'media':media,
            'runtime':read(root/'cloud/48h-extension-verification.json', None) or read(root/'cloud/20h-extension-verification.json', {})}
