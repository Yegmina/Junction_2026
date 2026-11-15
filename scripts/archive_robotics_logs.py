"""Continuously archive existing paid records; deliberately has no API client."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from aerorozum.cloud_cache import GCSRecordStore
from aerorozum.common import data_root, redacted_error
from aerorozum.label_archive import archive_pass

parser = argparse.ArgumentParser()
parser.add_argument('--once', action='store_true')
args = parser.parse_args()
root = data_root()
runs = ('robotics-er2-5fps', 'robotics-er2-airborne-priority', 'robotics-er2-airborne')


def sync(run):
    store = GCSRecordStore('aerorozum-jetson-testdata-smart-bird',
                           prefix='robotics-er2-labels/5fps-v1/audit/' + run)
    state = archive_pass(root/'labelling'/run/'labels', store,
                         root/'labelling'/run/'cloud-audit/state.json')
    return {'run': run, **{key: state[key] for key in
                          ('status', 'archived_versions', 'attempted', 'index_uri') if key in state},
            'failures': len(state['failures'])}


while True:
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(sync, run) for run in runs]:
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                print(json.dumps({'error': redacted_error(exc)}), flush=True)
    if args.once:
        break
    time.sleep(30)
