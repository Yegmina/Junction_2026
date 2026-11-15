import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('followup_plan', Path(__file__).resolve().parents[1] / 'cloud/stage_followup_label_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    rows = [{'id': str(i), 'sha256': str(i), 'source': {'generation': '1'}, 'airborne_eligible': True,
             'receipt_timestamp_ns': 100, 'eligibility': {'status': 'eligible', 'landed_state': 2,
                                                       'thermal_receipt_ns': 100, 'state_age_s': .2}} for i in range(3)]
    return {'policy': {'version': 'airborne-only-v2-primary-state'}, 'images': rows}, {'frames': copy.deepcopy(rows)}


def test_valid_and_invalid_paid_identities_are_both_excluded_from_new_calls():
    document, audit = inputs()
    records = [dict(document['images'][i], model=module.MODEL, valid_label=bool(i)) for i in range(2)]
    assert [r['id'] for r in module.pending_rows(document, audit, records)] == ['2']
    records[0]['source'] = {'generation': '2'}
    assert [r['id'] for r in module.pending_rows(document, audit, records)] == ['0', '2']


@pytest.mark.parametrize('mutation', [
    lambda row: row['eligibility'].update(landed_state=1),
    lambda row: row['eligibility'].update(state_age_s=3),
    lambda row: row.update(receipt_timestamp_ns=101),
    lambda row: row.update(receipt_timestamp_ns=None),
    lambda row: row.update(sha256='changed'),
])
def test_unbound_or_stale_frame_cannot_enter_paid_queue(mutation):
    document, audit = inputs()
    mutation(document['images'][0])
    with pytest.raises(ValueError):
        module.pending_rows(document, audit, [])
