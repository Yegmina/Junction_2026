import importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harbor/thermal-investigation-budgeted/environment'))
from camera_geometry import ground_to_pixel
spec=importlib.util.spec_from_file_location('budgeted_investigation',ROOT/'harbor/thermal-investigation-budgeted/environment/backend/server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)


@pytest.fixture
def task(tmp_path,monkeypatch):
    (tmp_path/'provenance').mkdir();(tmp_path/'sensor-scene').mkdir()
    (tmp_path/'sensor-scene/dawn-scene.json').write_text(json.dumps(dict(thermal_field_sha256='a'*64)))
    (tmp_path/'sensor-scene/scene.json').write_text(json.dumps(dict(thermal_field_sha256='a'*64,physics_time_hours=1.,task_thermal_phase='dawn')))
    (tmp_path/'provenance/scene.truth.json').write_text(json.dumps(dict(generated_together_with_insertions=True,
        field_hashes={'dawn':'a'*64},targets=[dict(id=0,x_m=8,y_m=0),dict(id=1,x_m=12,y_m=0),dict(id=2,x_m=0,y_m=12)])))
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.delenv('AEROROZUM_TASK_BUDGET',raising=False)
    state=dict(ready=True,sim_time_s=20.,pose=dict(x_m=0,y_m=0,z_m=.17),velocity={})
    monkeypatch.setattr(server,'native',lambda *args,**kwargs:state)
    instance=server.Investigation();instance.start=0;instance.end=15.;instance.home_return=True;instance.covered=set(range(200))
    return instance


def observation(identifier,z,x=8):
    return dict(id=identifier,pose=dict(x_m=x,y_m=0,z_m=z),width=640,height=512,intrinsics=dict(fx=400,fy=400),images=[],sim_time_s=2.)


def evidence(row,x=8,y=0):
    u,v=ground_to_pixel(row,x,y)
    return dict(observation_id=row['id'],bbox=dict(x_min=u-25,y_min=v-25,x_max=u+25,y_max=v+25))


def test_limits_and_close_evidence_requirements_are_public_before_takeoff(task):
    public=task.status()
    assert task.budget==75 and public['thermal_condition']['phase']=='dawn'
    assert public['evidence_requirements']==dict(max_inspection_altitude_m=4.5,multi_view_horizontal_baseline_m=1.)
    assert 'instance' not in public and 'targets' not in public


def test_wide_claim_cannot_earn_retrieval_credit_but_correct_close_evidence_can(task):
    wide=observation('wide',8);task.observations={'wide':wide}
    task.report=dict(candidates=[dict(evidence=[evidence(wide)])]);result=task.score()
    assert result['true_positives']==0 and result['false_positives']==1 and result['reward']==.2
    low=observation('close',4);task.observations['close']=low
    task.report['candidates'][0]['evidence'].append(evidence(low))
    assert task.score()['true_positives']==1 and task.score()['reward']>result['reward']


def test_inaccurate_close_box_cannot_borrow_correct_location_from_wide_view(task):
    wide=observation('wide',8);low=observation('close',4)
    task.observations={'wide':wide,'close':low}
    task.report=dict(candidates=[dict(evidence=[evidence(wide),evidence(low,x=9)])])
    # Median is within the retrieval tolerance, but the close image is not.
    assert task.score()['true_positives']==0


def test_vertical_only_revisit_does_not_earn_horizontal_view_bonus(task):
    wide=observation('wide',8);low=observation('close',4)
    task.observations={'wide':wide,'close':low}
    task.report=dict(candidates=[dict(evidence=[evidence(wide),evidence(low)])])
    assert task.score()['true_positives']==1 and task.score()['multi_view_matches']==0
    low['pose']['x_m']=9.5;task.report['candidates'][0]['evidence'][1]=evidence(low)
    assert task.score()['multi_view_matches']==1


def test_provisional_list_does_not_create_retrieval_credit_and_budget_remains_binding(task):
    low=observation('close',4);task.observations={'close':low}
    task.report=dict(candidates=[],provisional_candidates=[dict(evidence=[evidence(low)])])
    assert task.score()['true_positives']==0 and task.score()['false_positives']==0
    task.report['candidates']=task.report.pop('provisional_candidates');task.end=76
    assert task.score()['reward']==0
