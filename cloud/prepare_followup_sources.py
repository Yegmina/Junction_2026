"""Prepare newly discovered airborne thermal frames on the VM; no inference.

Original receipt integers and pinned object generations remain authoritative.
Sources without a one-to-one receipt clock are quarantined, never assumed flying.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

from aerorozum.common import digest, save, utcnow
from aerorozum.flight_filter import filter_rows, read_telemetry
from aerorozum.labelling import _probe, extract_video_samples, freeze_session_splits
from aerorozum.quarantine_recovery import receipt_records


def aligned_segment(records, segment, pts):
    local = [r for r in records if int(r.get('segment', 0)) == segment]
    if not pts or not all(math.isfinite(v) for v in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('Native presentation timestamps are not strictly increasing')
    if len(local) != len(pts):
        raise ValueError('Receipt/native frame count mismatch')
    times = [int(r['received_at_ns']) for r in local]
    if not times or times[0] <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Receipt timestamps are not strictly increasing')
    if not all('frame' in r for r in local):
        raise ValueError('Original receipt frame counters are required')
    if any(int(b['frame']) != int(a['frame']) + 1 for a, b in zip(local, local[1:])):
        raise ValueError('Receipt frame counters are not contiguous')
    return local


class PinnedObjects:
    def __init__(self, root, bucket, deadline):
        from google.auth.compute_engine import Credentials
        from google.auth.transport.requests import Request
        self.root, self.bucket, self.deadline = Path(root), bucket, deadline
        self.credentials, self.transport = Credentials(), Request()
        self.lock = threading.Lock()

    def get(self, url):
        if datetime.now(timezone.utc) >= self.deadline:
            raise RuntimeError('Preparation deadline reached')
        with self.lock:
            if not self.credentials.valid:
                self.credentials.refresh(self.transport)
            headers = {'Authorization': 'Bearer ' + self.credentials.token}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            return response.read()

    def object(self, name, generation=None):
        url = 'https://storage.googleapis.com/storage/v1/b/' + self.bucket + '/o/' + urllib.parse.quote(name, safe='')
        meta = json.loads(self.get(url + ('?generation=' + str(generation) if generation else '')))
        if generation is not None and str(meta['generation']) != str(generation):
            raise ValueError('Pinned object generation differs')
        folder = self.root / 'objects' / digest([self.bucket, name, str(meta['generation'])])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / Path(name).name
        if not path.exists():
            if shutil.disk_usage(self.root).free < int(meta['size']) + 10 * 1024**3:
                raise RuntimeError('Preserve 10 GiB free before downloading source objects')
            raw = self.get(url + '?alt=media&generation=' + str(meta['generation']))
            temporary = path.with_suffix('.partial')
            temporary.write_bytes(raw)
            if len(raw) != int(meta['size']) or (meta.get('md5Hash') and base64.b64encode(hashlib.md5(raw).digest()).decode() != meta['md5Hash']):
                temporary.unlink()
                raise ValueError('Pinned object size or MD5 differs')
            temporary.replace(path)
        raw = path.read_bytes()
        if len(raw) != int(meta['size']) or (meta.get('md5Hash') and base64.b64encode(hashlib.md5(raw).digest()).decode() != meta['md5Hash']):
            raise ValueError('Cached pinned object failed integrity verification')
        receipt = {'bucket': self.bucket, 'object': name, 'generation': str(meta['generation']),
                   'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
        save(folder / 'source.json', receipt)
        return path, receipt

    def inventory(self):
        url = 'https://storage.googleapis.com/storage/v1/b/' + self.bucket + '/o?prefix=jetson%2F&maxResults=1000'
        rows = []
        while url:
            document = json.loads(self.get(url))
            rows.extend(document.get('items', []))
            token = document.get('nextPageToken')
            url = url.split('&pageToken=')[0] + '&pageToken=' + urllib.parse.quote(token, safe='') if token else None
        return [r for r in rows if re.search(r'/recordings/[^/]+/thermal-\d+\.avi$', r['name'])]


def prepare(args):
    missing = [tool for tool in ['ffprobe', 'ffmpeg'] if shutil.which(tool) is None]
    if missing:
        raise RuntimeError('Preparation host dependencies missing: ' + ', '.join(missing))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.previous_audit.read_text())
    done = {(s['object'], str(s['generation'])) for s in previous['sources']}
    storage = PinnedObjects(root, args.bucket, datetime.fromisoformat(args.deadline_utc))
    inventory = storage.inventory()
    selected = [s for s in inventory if (s['name'], str(s['generation'])) not in done]
    if getattr(args,'only_sources',None):
        requested=json.loads(args.only_sources.read_text())
        allowed={(row['object'],str(row['generation'])) for row in requested['sources']}
        selected=[s for s in selected if (s['name'],str(s['generation'])) in allowed]
        if {(s['name'],str(s['generation'])) for s in selected}!=allowed:
            raise ValueError('A requested pinned source version is absent from the current inventory or was not released for requalification')
    save(root / 'inventory.json', {'at': utcnow(), 'normal_raw_sources': len(inventory),
                                 'sources': selected, 'previous_source_versions': len(done), 'paid_inference_issued': False})
    registry = root / 'session-splits.json'
    if not registry.exists():
        save(registry, json.loads(args.session_splits.read_text()))
    sessions = [re.search(r'/recordings/([^/]+)/', s['name']).group(1) for s in selected]
    splits = freeze_session_splits(sessions, registry, 17)

    def process(meta):
        source = {'bucket': args.bucket, 'object': meta['name'], 'generation': str(meta['generation'])}
        sid = re.search(r'/recordings/([^/]+)/', meta['name']).group(1)
        segment = int(re.search(r'thermal-(\d+)\.avi$', meta['name']).group(1))
        source.update(session_id=sid, segment=segment)
        folder = root / 'sources' / digest(source)[:20]
        folder.mkdir(parents=True, exist_ok=True)
        try:
            # Separate per-source clients avoid shared receipt writes in parallel.
            client = PinnedObjects(folder, args.bucket, storage.deadline)
            video, _ = client.object(meta['name'], meta['generation'])
            prefix = meta['name'].rsplit('/', 1)[0]
            receipt, receipt_identity = client.object(prefix + '/thermal-timestamps.jsonl')
            clean = folder / 'cleaned-receipts.jsonl'
            records, transform = receipt_records(receipt, clean)
            _, _, pts = _probe(video)
            local = aligned_segment(records, segment, pts)
            telemetry, telemetry_identity = client.object(prefix + '/telemetry.jsonl')
            parsed = read_telemetry(telemetry, telemetry_identity)
            rows = extract_video_samples(video, folder / 'frames', session_id=sid,
                                         source=source, timestamps_path=clean, segment=segment,
                                         fps=5, split=splits[sid], sample_origin_s=int(records[0]['received_at_ns']) / 1e9)
            for row in rows:
                row['receipt_timestamp_ns'] = int(local[row['frame_index']]['received_at_ns'])
                row['wallclock_alignment_available'] = True
                row['image'] = str(Path(row['image']).relative_to(root))
            audit, eligible = filter_rows(rows, {sid: parsed}, session_splits=splits)
            report = {**source, 'status': 'qualified', 'sampled_frames': len(rows),
                      'counts': audit['counts'], 'receipt': receipt_identity, 'receipt_transform': transform}
            save(folder / 'audit.json', audit)
            save(folder / 'eligible.json', eligible)
            save(folder / 'report.json', report)
            return report, audit['frames'], eligible['images']
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            report = {**source, 'status': 'quarantined', 'error_type': type(error).__name__,
                      'error': str(error)[:300], 'paid_inference_issued': False}
            save(folder / 'report.json', report)
            return report, [], []

    reports, frames, images = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for report, decisions, eligible in pool.map(process, selected):
            reports.append(report)
            frames.extend(decisions)
            images.extend(eligible)
            save(root / 'progress.json', {'at': utcnow(), 'sources_completed': len(reports),
                                        'sources_total': len(selected), 'eligible_frames': len(images),
                                        'quarantined_sources': sum(s['status'] == 'quarantined' for s in reports),
                                        'paid_inference_issued': False})
            print(json.dumps({'source': report['object'], 'status': report['status'], 'completed': len(reports)}), flush=True)
    from aerorozum.flight_filter import POLICY
    audit = {'schema_version': 1, 'at': utcnow(), 'policy': POLICY, 'sources': reports, 'frames': frames,
             'counts': {'total_frames': len(frames), 'airborne_eligible': len(images)}, 'paid_inference_issued': False}
    save(root / 'airborne-eligibility.json', audit)
    save(root / 'airborne-eligible.json', {'schema_version': 1, 'status': 'airborne_only', 'policy': POLICY,
         'images': images, 'session_splits': splits, 'eligibility_manifest': 'airborne-eligibility.json',
         'eligibility_manifest_sha256': hashlib.sha256((root / 'airborne-eligibility.json').read_bytes()).hexdigest(),
         'counts': audit['counts'], 'source_reports': reports, 'paid_inference_issued': False})
    save(root / 'result.json', {'at': utcnow(), 'status': 'prepared_with_quarantines' if any(s['status'] == 'quarantined' for s in reports) else 'prepared',
         'source_versions': len(selected), 'eligible_frames': len(images),
         'quarantined_sources': sum(s['status'] == 'quarantined' for s in reports),
         'paid_inference_issued': False, 'original_audit_modified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-audit', type=Path, required=True)
    parser.add_argument('--session-splits', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--bucket', default='aerorozum-jetson-testdata-smart-bird')
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--only-sources',type=Path,help='Optional frozen source-version selection for explicit requalification')
    args = parser.parse_args()
    if datetime.fromisoformat(args.deadline_utc).tzinfo is None:
        parser.error('An explicit UTC-offset deadline is required')
    prepare(args)
