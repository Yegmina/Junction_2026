import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('coverage_audit', Path(__file__).resolve().parents[1] / 'cloud/audit_label_coverage.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def frame(pixel, generation='1', state='eligible'):
    return {'sha256': pixel, 'source': {'generation': generation}, 'eligibility': {'status': state}}


def test_invalid_paid_response_cannot_become_a_negative_or_missing_request():
    audit = {'policy': {'version': 'airborne-only-v2-primary-state'},
             'frames': [frame('a'), frame('a'), frame('b'), frame('c'), frame('g', state='ground')]}
    records = [dict(frame('a'), model=audit_module.MODEL, valid_label=False),
               dict(frame('b'), model=audit_module.MODEL, valid_label=True),
               dict(frame('c', '2'), model=audit_module.MODEL, valid_label=True)]
    result = audit_module.reconcile(audit, records)
    assert result['unique_identity_counts'] == {'invalid_paid_response_retained': 1,
                                               'valid_label_retained': 1, 'no_paid_result_found': 1}
    assert result['frame_counts']['invalid_paid_response_retained'] == 2
    assert result['ground_frames_excluded'] == 1


def test_saved_success_is_preserved_and_discovery_is_reconciled_by_generation():
    audit = {'policy': {'version': 'airborne-only-v2-primary-state'}, 'frames': [frame('a')],
             'sources': [{'object': 'clip', 'generation': '1'}]}
    records = [dict(frame('a'), model=audit_module.MODEL, valid_label=v) for v in [True, False]]
    inventory = {'pending': [{'object': 'clip', 'generation': '1'}, {'object': 'clip', 'generation': '2'}]}
    result = audit_module.reconcile(audit, records, inventory)
    assert result['unique_identity_counts'] == {'valid_label_retained': 1}
    assert result['identities_with_both_valid_and_invalid_records'] == 1
    assert result['inventory_pending_already_in_frozen_audit'] == 1
    assert result['inventory_pending_outside_frozen_audit'] == 1
    with pytest.raises(ValueError, match='corrected airborne-only'):
        audit_module.reconcile(dict(audit, policy={'version': 'old'}), records)
