from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def data_root():
    root = Path(os.environ.get('AEROROZUM_DATA_ROOT', str(EXTERNAL / 'artifacts' if EXTERNAL.exists() else ROOT / 'artifacts')))
    root.mkdir(parents=True, exist_ok=True)
    return root


def config(path=None):
    return json.loads(Path(path or ROOT / 'configs/research.json').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False, default=str) + '\n'
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def secret(name, file_env=None):
    """Read an exact dotenv assignment; never execute or interpolate shell text."""
    if os.environ.get(name):
        return os.environ[name]
    if file_env and os.environ.get(file_env):
        return Path(os.environ[file_env]).read_text().strip()
    if name == 'GEMINI_API_KEY':
        private = Path.home() / '.config/aerorozum/gemini-api-key'
        if private.is_file():
            return private.read_text().strip()
    for path in (ROOT / '.env', ROOT.parent / '.env'):
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith('export '):
                    line = line[7:]
                key, sep, value = line.partition('=')
                if sep and key.strip() == name:
                    value = value.strip()
                    if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                        value = value[1:-1]
                    else:
                        value = value.split(' #', 1)[0].strip()
                    return value or None
    return None


def redacted_error(error):
    text = str(error)
    for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'NVIDIA_API_KEY'):
        value = secret(name, 'AEROROZUM_GEMINI_KEY_FILE' if name == 'GEMINI_API_KEY' else None)
        if value:
            text = text.replace(value, '[REDACTED]')
    return text[:2000]
