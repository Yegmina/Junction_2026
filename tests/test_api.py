from pathlib import Path
from fastapi.testclient import TestClient
from aerorozum.api import create_app


def client(tmp_path, evaluator=False):
    return TestClient(create_app(data_root=tmp_path,evaluator_enabled=evaluator))


def test_campaign_lifecycle_and_observation_media(tmp_path):
    c=client(tmp_path)
    created=c.post('/api/campaigns',json={'seed':17})
    assert created.status_code==200
    state=created.json(); identifier=state['id']
    assert state['mode']=='demo'
    assert 'targets' not in state
    launched=c.post(f'/api/campaigns/{identifier}/actions',json={'action':'launch','arguments':{}})
    assert launched.json()['ok']
    capture=c.post(f'/api/campaigns/{identifier}/actions',json={'action':'capture','arguments':{}}).json()
    assert capture['ok']
    obs=capture['observation']; path='campaigns/'+obs['image']
    response=c.get('/artifacts/'+path)
    assert response.status_code==200
    assert response.headers['content-type']=='image/png'
    def has_hidden(value):
        if isinstance(value,dict):return any(k in {'truth','targets','_targets','evaluator_snapshot'} or has_hidden(v) for k,v in value.items())
        if isinstance(value,list):return any(has_hidden(v) for v in value)
        return False
    assert not has_hidden(capture)


def test_evaluator_requires_server_and_request_opt_in(tmp_path):
    c=client(tmp_path)
    identifier=c.post('/api/campaigns',json={}).json()['id']
    assert c.get(f'/api/campaigns/{identifier}/evaluator',headers={'X-AeroRozum-Evaluator':'1'}).status_code==403
    c=client(tmp_path,evaluator=True)
    identifier=c.post('/api/campaigns',json={}).json()['id']
    assert c.get(f'/api/campaigns/{identifier}/evaluator').status_code==403
    assert c.get(f'/api/campaigns/{identifier}/evaluator',headers={'X-AeroRozum-Evaluator':'1'}).status_code==200


def test_private_paths_and_symlinks_are_not_served(tmp_path):
    c=client(tmp_path)
    (tmp_path/'observations').mkdir()
    outside=tmp_path.parent/'outside.png'; outside.write_bytes(b'private')
    (tmp_path/'observations'/'link.png').symlink_to(outside)
    (tmp_path/'observations'/'secret.json').write_text('{"targets":[]}')
    assert c.get('/artifacts/observations/link.png').status_code==403
    assert c.get('/artifacts/observations/secret.json').status_code==403
    assert c.get('/artifacts/private/observations/a.png').status_code==403


def test_status_redacts_secrets_and_does_not_invent_metrics(tmp_path):
    import json
    (tmp_path/'status.json').write_text(json.dumps({'providers':[{'id':'api','status':'not_configured','api_key':'secret-value'}]}))
    data=client(tmp_path).get('/api/status').json()
    assert data['training']['status']=='not_started'
    assert data['training']['metrics']==[]
    assert 'secret-value' not in str(data)

def test_overview_controls_cannot_accept_aircraft_commands(tmp_path):
    c=client(tmp_path)
    calls=[]
    class ViewOnly:
        def view(self,settings=None):
            calls.append(settings)
            return {'settings':settings or {'mode':'follow'},'confirmed':None}
    c.app.state.missions._bridge=ViewOnly()
    assert c.post('/api/missions/view',json={'distance_m':3.5,'mode':'follow'}).status_code==200
    assert calls==[{'mode':'follow','distance_m':3.5}]
    assert c.post('/api/missions/view',json={'drone_altitude_m':25}).status_code==422
    assert c.post('/api/missions/view',json={'distance_m':'Infinity'}).status_code==422
    assert len(calls)==1


def test_saved_candidate_examples_exclude_paid_provider_records(tmp_path):
    import json
    folder=tmp_path/'labelling/robotics-er2-airborne-priority/labels'
    folder.mkdir(parents=True)
    (folder/'positives.json').write_text(json.dumps({'images':[{'id':'candidate',
        'split':'train','image':'labelling/frames/frame.jpg','boxes':[{'confidence':.8}],
        'raw_response':{'prompt':'original private provider log'},'model_output':{'private':'log'}}]}))
    data=client(tmp_path).get('/api/labels/recent').json()['accepted_examples']
    assert len(data)==1 and data[0]['channel']=='saved_candidate_example'
    assert 'raw_response' not in data[0] and 'model_output' not in data[0]


def test_external_agent_campaigns_only_expose_public_state(tmp_path):
    import json
    run=tmp_path/'campaigns'/'agent-run';media=run/'artifacts'/'opaque'/'observations'/'frame.png'
    media.parent.mkdir(parents=True);media.write_bytes(b'png')
    public={'id':'opaque','pose':{'x_m':1,'y_m':2,'altitude_m':4},'observations':[{'id':'obs','image':'opaque/observations/frame.png'}]}
    (run/'state.json').write_text(json.dumps({'model':'teacher','status':'running','campaign_config':{'secret':'private'},'trajectory':[{'action':'capture','result':{'status':public}}]}))
    c=client(tmp_path);states=c.get('/api/status').json()['campaigns']
    assert states[0]['read_only'] and states[0]['controller']=='teacher'
    assert states[0]['observations'][0]['image']=='campaigns/agent-run/artifacts/opaque/observations/frame.png'
    assert 'private' not in str(states) and 'campaign_config' not in str(states)
    assert c.get('/api/campaigns/opaque').status_code==200


def test_internal_private_symlink_is_not_public(tmp_path):
    c=client(tmp_path);(tmp_path/'private').mkdir();(tmp_path/'observations').mkdir()
    private=tmp_path/'private'/'truth.png';private.write_bytes(b'private')
    (tmp_path/'observations'/'alias.png').symlink_to(private)
    assert c.get('/artifacts/observations/alias.png').status_code==403
