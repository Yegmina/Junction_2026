"""Reconcile saved ER2 results against frozen airborne eligibility, without APIs.

Input trees are read-only. Compact exports allow a VM cache to be reconciled
with an offline local cache without copying model responses or credentials.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

MODEL = 'gemini-robotics-er-2-preview'
CHANNELS = ('robotics-er2-5fps', 'robotics-er2-airborne-priority', 'robotics-er2-airborne')


def identity(row):
    source = row.get('source', {})
    if not row.get('sha256') or source.get('generation') is None:
        raise ValueError('A pixel hash and pinned source generation are required')
    return row['sha256'], str(source['generation'])


def compact_records(root):
    for channel in CHANNELS:
        for path in (Path(root) / 'labelling' / channel / 'labels/results').glob('*.json'):
            row = json.loads(path.read_text())
            if row.get('model') != MODEL:
                continue
            pixel, generation = identity(row)
            valid = (row.get('status') != 'invalid_model_response'
                     and isinstance(row.get('detections'), list)
                     and isinstance(row.get('zero_candidates'), bool))
            yield {'sha256': pixel, 'source': {'generation': generation},
                   'model': MODEL, 'valid_label': valid,
                   'status': row.get('status'), 'cache_key': row.get('cache_key', path.stem),
                   'channel': channel}


def reconcile(audit, records, inventory=None):
    if audit.get('policy', {}).get('version') != 'airborne-only-v2-primary-state':
        raise ValueError('The corrected airborne-only policy is required')
    cache = defaultdict(set)
    saved = 0
    for row in records:
        if row.get('model') != MODEL:
            continue
        cache[identity(row)].add('valid' if row['valid_label'] else 'invalid')
        saved += 1
    eligible = [f for f in audit['frames'] if f['eligibility']['status'] == 'eligible']
    decisions = {}
    for frame in eligible:
        key = identity(frame)
        states = cache.get(key, set())
        decisions[key] = ('valid_label_retained' if 'valid' in states else
                          'invalid_paid_response_retained' if states else 'no_paid_result_found')
    counts = Counter(decisions.values())
    processed = {(s['object'], str(s['generation'])) for s in audit.get('sources', [])}
    discovered_pending = {(s['object'], str(s['generation'])) for s in (inventory or {}).get('pending', [])}
    return {'schema_version': 1, 'at': datetime.now(timezone.utc).isoformat(),
            'status': 'cache_reconciled_not_all_sources_complete',
            'model': MODEL, 'policy_version': audit['policy']['version'],
            'eligible_frames': len(eligible), 'eligible_unique_identities': len(decisions),
            'unique_identity_counts': dict(counts),
            'frame_counts': dict(Counter(decisions[identity(f)] for f in eligible)),
            'saved_result_records_read': saved, 'saved_unique_identities': len(cache),
            'identities_with_both_valid_and_invalid_records': sum(len(v) > 1 for v in cache.values()),
            'frozen_source_versions': len(processed),
            'inventory_observed_at': (inventory or {}).get('at'),
            'inventory_pending_versions': len(discovered_pending),
            'inventory_pending_already_in_frozen_audit': len(discovered_pending & processed),
            'inventory_pending_outside_frozen_audit': len(discovered_pending - processed),
            'ground_frames_excluded': sum(f['eligibility']['status'] == 'ground' for f in audit['frames']),
            'unknown_frames_excluded': sum(f['eligibility']['status'] == 'unknown' for f in audit['frames']),
            'paid_inference_issued': False, 'input_records_modified': False,
            'limitations': ['Coverage is agreement with saved cache identities, not label accuracy.',
                           'Invalid paid responses remain invalid and are not negative labels.',
                           'Additional source versions require sampling and telemetry qualification.',
                           'Inventory is a timestamped snapshot, not a fresh bucket listing.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, action='append', default=[])
    parser.add_argument('--extra-index', type=Path, action='append', default=[])
    parser.add_argument('--export-index', type=Path)
    parser.add_argument('--eligibility', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [record for root in args.source_root for record in compact_records(root)]
    for path in args.extra_index:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if args.export_index:
        args.export_index.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in records))
    if args.eligibility:
        raw = args.eligibility.read_bytes()
        report = reconcile(json.loads(raw), records,
                           json.loads(args.inventory.read_text()) if args.inventory else None)
        report['eligibility_sha256'] = hashlib.sha256(raw).hexdigest()
    else:
        report = {'status': 'compact_cache_index_exported', 'records': len(records),
                  'paid_inference_issued': False}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
