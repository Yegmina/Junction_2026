from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from aerorozum.api import create_app
from aerorozum.common import save


def test_live_pipeline_reports_measurement_and_stale_data_separately(tmp_path):
    client = TestClient(create_app(tmp_path))
    fresh = {'telemetry': {'checked_at':datetime.now(timezone.utc).isoformat(),
                           'cpu_percent':33.5, 'gpus':[{'index':2,'utilization_percent':33}]},
             'controller':{'status':'waiting_for_accepted_physics'},
             'physics':{'accepted_pairs':0,'research_accepted':False}}
    save(tmp_path/'pipeline/worker.json', fresh)
    data = client.get('/api/pipeline').json()
    assert data['worker_connected'] and data['telemetry']['cpu_percent']==33.5
    assert data['controller']['status']=='waiting_for_accepted_physics'
    assert not data['physics']['research_accepted']
    fresh['telemetry']['checked_at']=(datetime.now(timezone.utc)-timedelta(minutes=3)).isoformat()
    save(tmp_path/'pipeline/worker.json', fresh)
    data = client.get('/api/pipeline').json()
    assert not data['worker_connected'] and data['telemetry']['cpu_percent']==33.5
    assert data['telemetry_age_s']>170


def test_pipeline_only_serves_published_image_media_and_blocks_symlinks(tmp_path):
    client = TestClient(create_app(tmp_path))
    media = tmp_path/'pipeline/media/physics';media.mkdir(parents=True)
    (media/'surface.png').write_bytes(b'actual image')
    private=tmp_path/'private';private.mkdir();(private/'truth.png').write_bytes(b'private')
    (media/'alias.png').symlink_to(private/'truth.png')
    save(tmp_path/'pipeline/status.json', {'secret':'private'})
    save(tmp_path/'pipeline/physics.json', {'media':[{'image':'pipeline/media/physics/surface.png'},
                                                    {'image':'private/truth.png'}]})
    assert len(client.get('/api/pipeline').json()['media'])==1
    assert client.get('/artifacts/pipeline/media/physics/surface.png').content==b'actual image'
    assert client.get('/artifacts/pipeline/media/physics/alias.png').status_code==403
    assert client.get('/artifacts/pipeline/status.json').status_code==403


def test_absent_or_malformed_worker_is_not_reported_connected(tmp_path):
    client=TestClient(create_app(tmp_path))
    assert not client.get('/api/pipeline').json()['worker_connected']
    save(tmp_path/'pipeline/worker.json',{'telemetry':{'checked_at':17}})
    assert not client.get('/api/pipeline').json()['worker_connected']


def test_live_view_returns_input_counts_without_transmitting_training_rows(tmp_path):
    client=TestClient(create_app(tmp_path))
    save(tmp_path/'pipeline/inputs/real.json',{'counts':{'images':8763},
         'images':[{'image':'private/frame.png','boxes':[1]}],'session_splits':{'private':'train'}})
    summary=client.get('/api/pipeline').json()['inputs']
    assert summary['counts']['images']==8763
    assert 'images' not in summary and 'session_splits' not in summary
    save(tmp_path/'pipeline/inputs/real.json',{'counts':{'images':8800},'images':[]})
    assert client.get('/api/pipeline').json()['inputs']['counts']['images']==8800


def test_recorded_training_media_is_never_described_as_synthetic(tmp_path):
    client=TestClient(create_app(tmp_path));media=tmp_path/'pipeline/media';media.mkdir(parents=True)
    (media/'real.jpg').write_bytes(b'actual recorded pixels')
    save(tmp_path/'pipeline/status.json',{'media':[{'image':'pipeline/media/real.jpg','provenance':'user_accepted_model'}]})
    assert client.get('/api/pipeline').json()['media'][0]['kind']=='training'
