import json
from cloud.overnight_missions import terminal_result


def test_harbor_initial_result_file_is_not_a_completed_trial(tmp_path):
    path=tmp_path/'result.json'
    path.write_text(json.dumps(dict(finished_at=None,stats={'n_running_trials':1})))
    assert terminal_result(path) is None
    path.write_text(json.dumps(dict(finished_at='2026-11-15T01:00:00+00:00',stats={'n_running_trials':0,'n_pending_trials':0})))
    assert terminal_result(path)['finished_at']
