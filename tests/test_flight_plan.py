import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'harbor/thermal-investigation-budgeted/environment/flight_plan.py'
spec = importlib.util.spec_from_file_location('flight_plan', MODULE)
flight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flight)


def test_invalid_plan_never_launches(tmp_path):
    calls = []
    with pytest.raises(ValueError):
        flight.collect({'waypoints': [{'x': 99, 'y': 1, 'altitude': 4}]}, tmp_path/'index.json',
                       call=lambda body: calls.append(body))
    assert calls == []


def test_return_reserve_skips_unaffordable_waypoint_and_lands(tmp_path):
    calls = []
    state = dict(pose=dict(x_m=0, y_m=0, z_m=2), remaining_s=25,
                 flight_elapsed_s=0, action_count=0)
    def call(body):
        calls.append(body['command'])
        return state.copy()
    result = flight.collect({'waypoints': [{'x': 24, 'y': 24, 'altitude': 4}]},
                            tmp_path/'index.json', call=call)
    assert calls == ['status', 'launch', 'status', 'land']
    assert result['status'] == 'landed' and result['observations'] == 0


def test_command_failure_still_returns_and_preserves_receipt(tmp_path):
    calls = []
    state = dict(pose=dict(x_m=0, y_m=0, z_m=2), remaining_s=75,
                 flight_elapsed_s=0, action_count=0)
    def call(body):
        calls.append(body['command'])
        if body['command'] == 'goto':
            raise RuntimeError('navigation unavailable')
        return state.copy()
    output = tmp_path/'index.json'
    result = flight.collect({'waypoints': [{'x': 2, 'y': 0, 'altitude': 4}]}, output, call=call)
    assert calls[-1] == 'land'
    assert result['errors'] == ['RuntimeError: navigation unavailable']
    assert output.exists()


def test_capture_and_landing_receipt_are_saved(tmp_path):
    calls = []
    state = dict(pose=dict(x_m=2, y_m=0, z_m=4), remaining_s=65,
                 flight_elapsed_s=0, action_count=0)
    def call(body):
        calls.append(body['command'])
        if body['command'] == 'observe':
            return dict(observation_id='actual-capture', pose=state['pose'], images=[], status=state.copy())
        return state.copy()
    result = flight.collect({'waypoints': [{'x': 2, 'y': 0, 'altitude': 4}]}, tmp_path/'index.json', call=call)
    assert result['observations'] == 1 and result['status'] == 'landed'
    assert calls[-1] == 'land'


def test_budget_dropping_during_navigation_triggers_return(tmp_path):
    calls = []
    state = dict(pose=dict(x_m=0, y_m=0, z_m=2), remaining_s=75,
                 flight_elapsed_s=0, action_count=0)
    def call(body):
        calls.append(body['command'])
        if body['command'] == 'goto':
            state.update(remaining_s=8, flight_elapsed_s=67)
        return state.copy()
    result = flight.collect({'waypoints': [{'x': 2, 'y': 0, 'altitude': 4}]}, tmp_path/'index.json', call=call)
    assert result['status'] == 'landed'
    assert calls[-1] == 'land' and 'observe' not in calls


def test_image_downloads_happen_after_landing(tmp_path, monkeypatch):
    import io
    calls = []
    state = dict(pose=dict(x_m=2, y_m=0, z_m=4), remaining_s=65,
                 flight_elapsed_s=0, action_count=0)
    def call(body):
        calls.append(body['command'])
        if body['command'] == 'observe':
            return dict(observation_id='actual', images=[dict(filename='actual-thermal.png', url='/evidence/actual-thermal.png')])
        return state.copy()
    def download(url, timeout):
        assert calls[-1] == 'land'
        return io.BytesIO(b'paired-image-bytes')
    monkeypatch.setattr(flight.urllib.request, 'urlopen', download)
    result = flight.collect({'waypoints': [{'x': 2, 'y': 0, 'altitude': 4}]}, tmp_path/'index.json', call=call)
    assert result['errors'] == []
    assert (tmp_path/'actual-thermal.png').read_bytes() == b'paired-image-bytes'
