"""Submission-level reward and state attacks, independent of model outputs."""
import importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'harbor/thermal-investigation/environment'))
spec=importlib.util.spec_from_file_location('harbor_investigation',ROOT/'harbor/thermal-investigation/environment/backend/server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)

@pytest.fixture
def task(tmp_path,monkeypatch):
    (tmp_path/'provenance').mkdir();(tmp_path/'sensor-scene').mkdir()
    (tmp_path/'sensor-scene/warming-scene.json').write_text(json.dumps({'thermal_field_sha256':'a'*64}))
    (tmp_path/'sensor-scene/scene.json').write_text(json.dumps({'physics_time_hours':7.5,'thermal_field_sha256':'a'*64}))
    (tmp_path/'provenance/scene.truth.json').write_text(json.dumps({'generated_together_with_insertions':True,'field_hashes':{'warming':'a'*64},'targets':[{'id':0,'x_m':8,'y_m':0},{'id':1,'x_m':12,'y_m':0},{'id':2,'x_m':0,'y_m':12}]}))
    monkeypatch.setattr(server,'ROOT',tmp_path)
    state=dict(ready=True,sim_time_s=20.,pose=dict(x_m=0,y_m=0,z_m=.17),velocity=dict(x_m_s=0,y_m_s=0,z_m_s=0))
    monkeypatch.setattr(server,'native',lambda *args,**kwargs:state)
    return server.Investigation(),state

def row(x=8):
    return dict(id='real',pose=dict(x_m=x,y_m=0,z_m=5.12),width=640,height=512,intrinsics=dict(fx=400,fy=400),images=[],sim_time_s=2)

def candidate(obs='real'):
    return dict(evidence=[dict(observation_id=obs,bbox=dict(x_min=450,y_min=450,x_max=550,y_max=550))])

def ready(task):
    t,state=task;t.start=0;t.end=15.;t.home_return=True;t.observations={'real':row()};return t,state

def test_noop_and_early_seal_score_zero(task):
    t,state=task
    assert t.seal()['reward']==0
    with pytest.raises(ValueError,match='sealed'):t.tool({'command':'observe'})

def test_empty_evidence_and_forged_observation_ids_cannot_claim_objects(task):
    t,state=ready(task);t.report={'candidates':[candidate('forged')]}
    score=t.score();assert score['true_positives']==0 and score['invalid_candidates']==1 and score['reward']==0

def test_actual_ray_matches_one_target_and_duplicate_claims_reduce_score(task):
    t,state=ready(task);t.report={'candidates':[candidate()]};one=t.score()
    assert one['true_positives']==1 and one['reward']>0
    t.report['candidates'].append(candidate());duplicate=t.score()
    assert duplicate['true_positives']==1 and duplicate['false_positives']==1 and duplicate['reward']<one['reward']

def test_repeated_same_capture_is_not_independent_view_evidence(task):
    t,state=ready(task);c=candidate();c['evidence']*=3;t.report={'candidates':[c]}
    assert t.score()['multi_view_matches']==0


def test_copying_region_wide_guesses_into_every_record_cannot_retrieve_targets(task):
    t,state=ready(task)
    t.observations['other']={**row(12),'id':'other'}
    mixed=candidate();mixed['evidence'].append(candidate('other')['evidence'][0])
    t.report={'candidates':[mixed,mixed,mixed]}
    score=t.score()
    assert score['true_positives']==0 and score['invalid_candidates']==3


def test_different_boxes_in_one_capture_cannot_be_one_candidate(task):
    t,state=ready(task);mixed=candidate()
    mixed['evidence'].append(dict(observation_id='real',bbox=dict(x_min=550,y_min=450,x_max=650,y_max=550)))
    t.report={'candidates':[mixed]}
    assert t.score()['true_positives']==0 and t.score()['invalid_candidates']==1


def test_one_lucky_view_cannot_override_inconsistent_majority_location(task):
    t,state=ready(task)
    t.observations.update(other={**row(8.8),'id':'other'},third={**row(8.9),'id':'third'})
    c=candidate();c['evidence'] += [candidate('other')['evidence'][0],candidate('third')['evidence'][0]]
    t.report={'candidates':[c]}
    assert t.score()['true_positives']==0 and t.score()['invalid_candidates']==0

def test_ground_control_and_budget_are_completion_gates(task):
    t,state=ready(task);t.report={'candidates':[candidate()]};assert t.score()['reward']>0
    t.end=181.;assert t.score()['reward']==0
    t.end=15.;state['pose']['z_m']=3.;assert t.score()['reward']==0

def test_full_image_boxes_do_not_score(task):
    t,state=ready(task);c=candidate();c['evidence'][0]['bbox']=dict(x_min=0,y_min=0,x_max=1000,y_max=1000);t.report={'candidates':[c]}
    assert t.score()['true_positives']==0 and t.score()['invalid_candidates']==1

def test_landing_freezes_flight_clock_without_allowing_more_flight(task):
    t,state=ready(task);state['sim_time_s']=1000.;assert t.elapsed(state)==15.
    with pytest.raises(ValueError,match='flight has ended'):t.tool(dict(command='goto',x=8,y=0,altitude=5))

def test_coverage_uses_actual_captured_polygon():
    polygon=[server.project(row(),u,v) for u,v in [(0,0),(1000,0),(1000,1000),(0,1000)]]
    assert server.inside((8,0),polygon)
    assert not server.inside((0,12),polygon)


def test_private_instance_and_reference_are_not_in_public_status(task):
    t,_=task
    public=t.status()
    assert 'instance' not in public and 'targets' not in public and 'seed' not in public
    assert public['thermal_condition']['phase']=='warming'


def test_reference_identity_must_match_camera_fields(task,tmp_path):
    (tmp_path/'sensor-scene/warming-scene.json').write_text(json.dumps({'thermal_field_sha256':'b'*64}))
    with pytest.raises(ValueError,match='identities differ'):server.Investigation()


def test_public_condition_preserves_the_selected_physical_snapshot(task,tmp_path):
    (tmp_path/'sensor-scene/scene.json').write_text(json.dumps(dict(physics_time_hours=1.,task_thermal_phase='dawn',thermal_field_sha256='a'*64)))
    state=server.Investigation().status()
    assert state['thermal_condition']==dict(phase='dawn',physics_elapsed_hours=1.)
    assert 'seed' not in state


def test_active_camera_cannot_point_to_an_unreferenced_field(task,tmp_path):
    (tmp_path/'sensor-scene/scene.json').write_text(json.dumps(dict(physics_time_hours=7.5,thermal_field_sha256='c'*64)))
    with pytest.raises(ValueError,match='Active camera field'):server.Investigation()


@pytest.mark.parametrize('u,v',[(200,300),(700,600),(500,500)])
def test_public_camera_calibration_round_trips_ground_coordinates(u,v):
    from camera_geometry import pixel_to_ground,ground_to_pixel
    observation=row();observation['pose'].update(roll_rad=.1,pitch_rad=.2,yaw_rad=.7)
    x,y=pixel_to_ground(observation,u,v)
    assert ground_to_pixel(observation,x,y)==pytest.approx((u,v),abs=1e-8)
