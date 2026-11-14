"""Refresh actual artifact-derived state; never fabricate model/physics progress."""
import time
from aerorozum.common import data_root,save,redacted_error,utcnow
from aerorozum.preflight import snapshot

while True:
    try:snapshot()
    except Exception as ex:save(data_root()/'monitor-error.json',dict(at=utcnow(),error=redacted_error(ex)))
    time.sleep(15)
