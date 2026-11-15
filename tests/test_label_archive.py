import json

from aerorozum.common import save
from aerorozum.label_archive import archive_pass


class Store:
    def __init__(self):
        self.objects = {}
        self.fail = False
        self.calls = 0

    def put(self, key, record, kind='results'):
        if self.fail:
            raise RuntimeError('network unavailable')
        self.calls += 1
        self.objects.setdefault((kind, key), record)
        return 'gs://test/' + kind + '/' + key


def test_archives_truncated_paid_attempt_and_errors_without_inference(tmp_path):
    folder = tmp_path/'labels'
    original = {'raw_text': '{"detections": [', 'usage': {'total_token_count': 74}}
    save(folder/'provider/raw/attempt-0.json', original)
    save(folder/'provider/errors/attempt-0.json', {'error': 'truncated JSON'})
    store = Store()
    state = archive_pass(folder, store, tmp_path/'state.json')
    assert state['status'] == 'synced' and state['archived_versions'] == 2
    assert any(value.get('original') == original for value in store.objects.values())
    again = archive_pass(folder, store, tmp_path/'state.json')
    assert again['attempted'] == 0 and again['archived_versions'] == 2


def test_failed_upload_is_retried_and_changed_record_keeps_old_version(tmp_path):
    folder = tmp_path/'labels'
    save(folder/'progress.json', {'completed': 1})
    store = Store(); store.fail = True
    state = archive_pass(folder, store, tmp_path/'state.json')
    assert state['status'] == 'upload_pending' and not state['records']
    store.fail = False
    assert archive_pass(folder, store, tmp_path/'state.json')['archived_versions'] == 1
    save(folder/'progress.json', {'completed': 2})
    state = archive_pass(folder, store, tmp_path/'state.json')
    assert state['archived_versions'] == 2
    originals = [value['original'] for (kind, _), value in store.objects.items() if kind == 'logs']
    assert {'completed': 1} in originals and {'completed': 2} in originals
