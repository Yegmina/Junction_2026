import copy

import pytest

from aerorozum.vm_ingestion import join_flight_audits, merge_catalogs


def row(name, split='train', zero=True):
    return dict(id=name, sha256=name, split=split, image=name + '.jpg', boxes=[] if zero else [{'bbox': [1, 2, 3, 4]}],
                provenance='user_accepted_model', row_kind='native_label', source_group='video:session',
                source=dict(object='clip', generation='1', session_id='session'), frame_index=int(name),
                zero_candidates=zero, detections=[] if zero else [{'confidence': .9}],
                image_statistics=dict(gray_stddev=8, entropy_bits=5, width=640, height=512))


def audit(rows):
    return dict(policy={'version': 'airborne-only-v2-primary-state'}, sources=[{'object': 'clip', 'generation': '1'}],
                frames=[dict(r, eligibility={'status': 'eligible'}) for r in rows])


def test_merge_retains_anchors_and_protects_test_hashes_and_ground_frames():
    rows = [row('1'), row('2'), row('3'), row('4', split='test')]
    anchor = dict(row('5'), row_kind='manual_anchor', provenance='original_manual')
    source = audit(rows);source['frames'][1]['eligibility']['status'] = 'ground'
    flight = join_flight_audits([source])
    base = dict(images=[anchor], sealed_test_image_hashes=['1'], session_splits={'session': 'train'})
    before = copy.deepcopy(base)
    merged, backgrounds = merge_catalogs(base, {'images': rows}, flight)
    by_id = {r['id']: r for r in merged['images']}
    assert by_id['1']['training_eligible'] is False
    assert by_id['2']['training_eligible'] is False and by_id['2']['flight_state'] == 'known_ground'
    assert '4' not in by_id and '4' in merged['sealed_test_image_hashes']
    assert by_id['5']['provenance'] == 'original_manual'
    assert all(r['id'] not in ['1', '2', '4'] for r in backgrounds['crops'])
    assert backgrounds['target_free_verified'] is False and base == before


def test_conflicting_flight_or_label_bindings_are_never_silently_overwritten():
    first = audit([row('1')]);second = copy.deepcopy(first)
    second['frames'][0]['eligibility']['status'] = 'ground'
    with pytest.raises(ValueError, match='Conflicting frozen flight'):
        join_flight_audits([first, second])
    changed = row('1', zero=False)
    with pytest.raises(ValueError, match='conflicting labels'):
        merge_catalogs({'images': [row('1')]}, {'images': [changed]}, join_flight_audits([first]))
    with pytest.raises(ValueError, match='session split'):
        merge_catalogs({'session_splits': {'session': 'train'}}, {'session_splits': {'session': 'validation'}}, join_flight_audits([first]))


def test_validation_duplicate_cannot_become_training_background():
    train = row('1');validation = dict(row('2', split='validation'), sha256='1')
    merged, backgrounds = merge_catalogs({'images': [train]}, {'images': [validation]}, join_flight_audits([audit([train, validation])]))
    assert all(r['cross_split_duplicate'] for r in merged['images'])
    assert not backgrounds['crops']
