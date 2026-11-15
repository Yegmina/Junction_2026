"""Deterministic training-only expansion with equal positive/zero-frame counts."""
from collections import defaultdict
from .common import digest


def select(catalog, development_hashes=(), seed=17):
    held = set(catalog.get('sealed_test_image_hashes', ())) | set(development_hashes)
    held.update(r['sha256'] for r in catalog['images'] if r['split'] != 'train')
    unique = {}
    for row in catalog['images']:
        if row['split'] != 'train' or not row.get('training_eligible') or row.get('cross_split_duplicate'):
            continue
        if row['sha256'] in held or row.get('training_excluded_reason') or row.get('uncertain_boxes'):
            continue
        if row.get('provenance') not in ('original_manual', 'reviewed_ai', 'user_accepted_model'):
            continue
        if row.get('provenance') == 'user_accepted_model' and row.get('flight_state') != 'airborne':
            continue
        unique.setdefault(row['sha256'], row)
    positives = [r for r in unique.values() if r.get('boxes')]
    groups = defaultdict(list)
    for row in unique.values():
        if not row.get('boxes') and row.get('zero_candidates') is True and row.get('detections') == []:
            groups[row['source_group']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: digest([seed, r['sha256']]))
    negatives = []
    while groups and len(negatives) < len(positives):
        for key in sorted(list(groups)):
            negatives.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(negatives) == len(positives):
                break
    if not positives or len(negatives) != len(positives):
        raise ValueError('Balanced training needs positive evidence and enough explicit zero-candidate frames')
    rows = sorted(positives + negatives, key=lambda r: digest([seed, 'order', r['sha256']]))
    metadata = dict(namespace='incremental-stream-v1', frozen_stream_batch=True, status='passed',
                    batch_id=digest([catalog['dataset_signature'], seed, [r['sha256'] for r in rows]])[:24],
                    experiment_branch='expanded_visual_curriculum_v1', reward_protocol='soft-iou-f2-v2',
                    accepted_pair_certificates={}, native_positive_frames=len(positives),
                    explicit_zero_frames=len(negatives), all_eligible_positives_retained=True,
                    test_accessed=False, automatic_promotion=False,
                    interpretation='Training-only expansion; machine labels remain distinct from human labels. Zero-candidate frames are not verified mine-free ground.')
    return dict(metadata=metadata, images=rows, session_splits=catalog['session_splits'])
