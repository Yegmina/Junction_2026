"""Join VM label arrivals with the preserved catalog without weakening admission.

Builds a candidate catalog first. Publishing it to the trainer is a separate
operation so an experiment recipe cannot change midway through a run.
"""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

from .common import digest, save, utcnow
from .labelling import MODEL
from .real_ingester import RealIngester


def flight_key(row):
    source = row['source']
    return source['object'], str(source['generation']), row['frame_index']


def join_flight_audits(audits):
    frames, sources = {}, {}
    for audit in audits:
        if audit.get('policy', {}).get('version') != 'airborne-only-v2-primary-state':
            raise ValueError('Only the corrected airborne policy may enter the joined audit')
        for row in audit['frames']:
            key = flight_key(row)
            if key in frames and frames[key]['eligibility'] != row['eligibility']:
                raise ValueError('Conflicting frozen flight decisions require review')
            frames.setdefault(key, row)
        for source in audit.get('sources', []):
            sources.setdefault((source['object'], str(source['generation'])), source)
    if not audits:
        raise ValueError('A frozen flight audit is required')
    counts = Counter(r['eligibility']['status'] for r in frames.values())
    return dict(schema_version=1, status='joined_frozen_vm_audits', policy=audits[0]['policy'],
                frames=list(frames.values()), sources=list(sources.values()),
                counts=dict(total_frames=len(frames), airborne_eligible=counts['eligible'],
                            ground_excluded=counts['ground'], unknown_skipped=counts['unknown']),
                paid_inference_issued=False)


def merge_catalogs(base, incoming, flight):
    sessions = dict(base.get('session_splits', {}))
    for session, split in incoming.get('session_splits', {}).items():
        if session in sessions and sessions[session] != split:
            raise ValueError('A frozen session split changed')
        sessions[session] = split
    sealed = set(base.get('sealed_test_image_hashes', []))
    sealed.update(incoming.get('sealed_test_image_hashes', []))
    by_key = {}
    for original in base.get('images', []) + incoming.get('images', []):
        if original['split'] not in ('train', 'validation'):
            sealed.add(original['sha256'])
            continue
        row = copy.deepcopy(original)
        key = (row['id'], row['sha256'], row['split'])
        previous = by_key.get(key)
        if previous and (previous.get('boxes'), previous.get('provenance'), previous.get('source')) != (row.get('boxes'), row.get('provenance'), row.get('source')):
            raise ValueError('Same image identity has conflicting labels or provenance')
        by_key[key] = row
    splits = defaultdict(set)
    for row in by_key.values():
        splits[row['sha256']].add(row['split'])
    decisions = {flight_key(row): row['eligibility'] for row in flight['frames']}
    rows = sorted(by_key.values(), key=lambda r: (r.get('source_group', ''), r['id']))
    for row in rows:
        duplicate = row['sha256'] in sealed or len(splits[row['sha256']]) > 1
        row['cross_split_duplicate'] = duplicate
        row['training_eligible'] = row['split'] == 'train' and not duplicate
        if row.get('row_kind') == 'native_label':
            decision = decisions.get(flight_key(row))
            state = decision['status'] if decision else 'unknown'
            row['flight_state'] = {'eligible': 'airborne', 'ground': 'known_ground'}.get(state, 'unknown')
            row['flight_eligibility'] = decision or dict(status='unknown', reason='no_frozen_flight_binding', airborne_eligible=False)
            row['training_eligible'] &= state == 'eligible'
            if state != 'eligible':
                row['training_excluded_reason'] = 'airborne_only_policy_' + row['flight_state']
    background_groups = defaultdict(list)
    for row in rows:
        if not (row['training_eligible'] and row.get('zero_candidates') is True and row.get('detections') == []):
            continue
        stats = row['image_statistics']
        if row.get('row_kind') == 'native_label' and not (stats['gray_stddev'] >= 3 and stats['entropy_bits'] >= 2 and stats['width'] >= 128 and stats['height'] >= 96):
            continue
        background = dict(row, screening=dict(status='accepted', decision='no_mine_detected', model=MODEL,
                          detections=[], rule='zero_candidates_at_any_confidence', human_reviewed=False, verified_negative=False),
                          target_free_verified=False, background_identity_status='machine_screened_no_candidate_not_verified_negative',
                          background_source='annotation_excluded_crop' if row['row_kind'] == 'zip_screened_background' else 'bucket_frame_explicit_zero_candidates')
        background_groups[row['source_group']].append(background)
    backgrounds = []
    while background_groups:
        for key in sorted(list(background_groups)):
            backgrounds.append(background_groups[key].pop())
            if not background_groups[key]:
                del background_groups[key]
    counts = dict(images=len(rows), sealed_test_unique_images=len(sealed),
                  native_labels=sum(r.get('row_kind') == 'native_label' for r in rows),
                  manual_anchor_train=sum(r.get('row_kind') == 'manual_anchor' and r['training_eligible'] for r in rows),
                  zip_screened_backgrounds=sum(r.get('row_kind') == 'zip_screened_background' for r in rows),
                  train_positive_frames=sum(r['training_eligible'] and bool(r.get('boxes')) for r in rows),
                  train_zero_frames=sum(r['training_eligible'] and r.get('zero_candidates') is True for r in rows),
                  cross_split_duplicate_rows=sum(r['cross_split_duplicate'] for r in rows), background_bank=len(backgrounds),
                  splits=dict(Counter(r['split'] for r in rows)))
    counts.update({'native_' + state: sum(r.get('row_kind') == 'native_label' and r.get('flight_state') == state for r in rows)
                   for state in ['airborne', 'known_ground', 'unknown']})
    result = dict(schema_version=1, updated_at=utcnow(), images=rows, counts=counts, session_splits=sessions,
                  zone_splits=base.get('zone_splits', {}), sealed_test_image_hashes=sorted(sealed),
                  airborne_only_required=True, transfer_scope='Training and development only; test labels and paths excluded',
                  flight_filter=dict(policy=flight['policy'], counts=flight['counts'], paid_inference_issued=False))
    result['dataset_signature'] = digest({'images': [(r['id'], r['sha256'], r['split'], r.get('boxes'), r['training_eligible']) for r in rows],
                                          'sealed_test_image_hashes': sorted(sealed), 'session_splits': sessions})
    bank = dict(schema_version=2, image_root='artifact_root', status='screened', updated_at=result['updated_at'],
                screening_model=MODEL, accepted_crops=len(backgrounds), crops=backgrounds,
                target_free_verified=False, allowed_splits=['train'],
                limitation='Explicit zero-candidate machine screening is not human-confirmed mine-free ground.')
    return result, bank


def build_candidate(labeler_root, trainer_root):
    labeler_root, trainer_root = Path(labeler_root).resolve(), Path(trainer_root).resolve()
    state = labeler_root.parent / 'state/launch.json'
    plan = json.loads(state.read_text())
    active_manifest = json.loads(Path(plan['manifest_path']).read_text())
    snapshots = list((labeler_root / 'pipeline/inputs/airborne-snapshots').glob('*/airborne-eligibility.json'))
    audits = [json.loads(path.read_text()) for path in snapshots]
    flight = join_flight_audits(audits)
    save(labeler_root / 'pipeline/inputs/airborne-eligibility.json', flight)
    registry = labeler_root / 'labelling/robotics-er2-airborne/session-splits.json'
    save(registry, dict(seed=17, sessions=active_manifest['session_splits']))
    reader = RealIngester(labeler_root)
    incoming = reader.scan()
    history = trainer_root / 'pipeline/inputs/vm-ingestion'
    history.mkdir(parents=True, exist_ok=True)
    baseline = history / 'base-before-vm-label-ingestion.json'
    if not baseline.exists():
        shutil.copyfile(trainer_root / 'pipeline/inputs/real.json', baseline)
    base = json.loads(baseline.read_text())
    candidate, bank = merge_catalogs(base, incoming, flight)
    verifier = RealIngester(trainer_root)
    for row in incoming['images']:
        if row['split'] not in ('train', 'validation'):
            continue
        source = (labeler_root / row['image']).resolve()
        target = (trainer_root / row['image']).resolve()
        if not source.is_relative_to(labeler_root) or not target.is_relative_to(trainer_root):
            raise ValueError('Arrival image escapes an artifact tree')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for row in candidate['images']:
        verifier.image(row['image'], row['sha256'])
    verifier.db.commit()
    save(history / 'candidate.json', candidate)
    save(history / 'backgrounds-candidate.json', bank)
    receipt = dict(status='candidate_pixels_verified_not_published_to_trainer', at=utcnow(),
                   dataset_signature=candidate['dataset_signature'], counts=candidate['counts'],
                   frozen_base_catalog=str(baseline), joined_source_versions=len(flight['sources']),
                   source_flight_counts=flight['counts'], incoming_accepted_rows=len(incoming['images']),
                   incoming_excluded_records=incoming['counts']['excluded_records'],
                   active_training_catalog_changed=False, paid_inference_issued=False)
    save(history / 'receipt.json', receipt)
    reader.db.close();verifier.db.close()
    return receipt
