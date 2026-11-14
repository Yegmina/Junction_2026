import copy
import hashlib
import json

import numpy as np
import pytest

from aerorozum.environment import Campaign, baseline_policy, certify_observability
from aerorozum.evaluation import score_submission, valid_box
from aerorozum.mcp_server import AgentTools


def make(tmp_path, **config):
    return Campaign({"camera":{"width":128,"height":128}, **config},seed=17,output_dir=tmp_path)


def test_six_sorties_and_turnaround(tmp_path):
    campaign = make(tmp_path)
    for index in range(6):
        if index:
            refused = campaign.act("launch")
            assert not refused["ok"] and "Turnaround" in refused["error"]
            assert campaign.act("wait_until",{"time_s":campaign.next_launch_s})["ok"]
        assert campaign.act("launch")["ok"]
        assert campaign.act("land")["ok"]
    assert not campaign.act("launch")["ok"]
    assert campaign.sorties_used == 6
    assert campaign.total_airborne_s > 0


def test_movement_capture_time_and_battery(tmp_path):
    campaign = make(tmp_path)
    assert campaign.act("launch",{"altitude_m":6})["ok"]
    start = campaign.status()
    moved = campaign.act("move_to",{"x_m":44,"y_m":32})
    assert moved["ok"]
    assert campaign.time_s == start["time_s"]+6
    before = campaign.status()["remaining_sortie_s"]
    captured = campaign.act("capture")
    assert captured["ok"]
    assert campaign.status()["remaining_sortie_s"] == before-2
    assert (tmp_path/captured["observation"]["image"]).is_file()
    assert captured["observation"]["temperature_measurement"] == "display_only_not_radiometric"


def test_airborne_wait_reserves_landing_and_ground_wait_costs_no_battery(tmp_path):
    campaign = make(tmp_path)
    campaign.act("launch")
    before = campaign.status()
    bad = campaign.act("wait_until",{"time_s":campaign.time_s+590})
    assert not bad["ok"] and "battery" in bad["error"]
    assert campaign.time_s == before["time_s"]
    assert campaign.act("land")["ok"]
    battery = campaign.total_airborne_s
    assert campaign.act("wait_until",{"time_s":campaign.time_s+10000})["ok"]
    assert campaign.total_airborne_s == battery


@pytest.mark.parametrize("action,args",[("move_to",{"x_m":9999}),
    ("move_to",{"altitude_m":float("nan")}), ("wait_until",{"time_s":0}),
    ("inspect_observation",{"observation_id":"../../etc/passwd"}),
    ("evaluator_snapshot",{}), ("capture",{})])
def test_invalid_actions_do_not_advance_time(tmp_path,action,args):
    campaign = make(tmp_path)
    before = campaign.time_s
    result = campaign.act(action,args)
    assert result["ok"] is False
    assert campaign.time_s == before


def test_private_truth_and_config_are_not_returned(tmp_path):
    config = {"campaign":{"width_m":64},"camera":{"width":128,"height":128},
              "_targets":[{"id":"SECRET-OBJECT-ID","x_m":32,"y_m":32,"radius_m":.5}]}
    original = copy.deepcopy(config)
    campaign = Campaign(config,seed=2,output_dir=tmp_path)
    campaign.act("launch")
    observed = campaign.act("capture")
    status = campaign.status()
    assert config == original
    status["pose"]["x_m"] = -10
    assert campaign.pose["x_m"] == 32
    assert "SECRET-OBJECT-ID" not in json.dumps(observed)
    assert "truth" not in observed["observation"]
    assert campaign.evaluator_snapshot()["targets"][0]["id"] == "SECRET-OBJECT-ID"
    assert not status["research_accepted"]


def test_replay_same_seed_same_sensor_pixels(tmp_path):
    hashes = []
    for directory in (tmp_path/"a",tmp_path/"b"):
        campaign = make(directory)
        campaign.act("launch",{"altitude_m":5})
        campaign.act("move_to",{"x_m":10,"y_m":12,"yaw_deg":35})
        result = campaign.act("capture")
        hashes.append(hashlib.sha256((directory/result["observation"]["image"]).read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_noop_and_empty_report_do_not_get_negative_scene_credit(tmp_path):
    campaign = make(tmp_path,demo_target_count=0)
    assert campaign.evaluator_snapshot()["score"]["reward"] == 0
    campaign.act("submit_report")
    assert campaign.evaluator_snapshot()["score"]["reward"] == 0


def test_repeat_capture_does_not_farm_coverage(tmp_path):
    campaign = make(tmp_path,demo_target_count=0,camera={"width":640,"height":512})
    campaign.act("launch")
    campaign.act("capture")
    coverage = campaign.evaluator_snapshot()["coverage"]
    assert coverage > 0
    campaign.act("capture")
    assert campaign.evaluator_snapshot()["coverage"] == coverage
    campaign.act("submit_report")
    assert campaign.evaluator_snapshot()["score"]["reward"] == coverage


def test_reward_duplicates_spam_and_unobserved_evidence():
    box = {"x_min":450,"y_min":450,"x_max":550,"y_max":550}
    observations = {"frame":{"truth":[{"target_id":"t1","box":box,"observable":True}]}}
    prediction = {"evidence":[{"observation_id":"frame","box":box}]}
    correct = score_submission([prediction],observations,{"t1"},1)
    duplicate = score_submission([prediction,prediction],observations,{"t1"},1)
    spam = {"evidence":[{"observation_id":"frame","box":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000}}]}
    wrong = score_submission([spam],observations,{"t1"},1)
    unknown = {"evidence":[{"observation_id":"invented","box":box}]}
    absent = score_submission([unknown],observations,{"t1"},1)
    assert correct["reward"] == 1
    assert duplicate["true_positives"] == 1 and duplicate["false_positives"] == 1
    assert duplicate["reward"] < correct["reward"]
    assert wrong["true_positives"] == absent["true_positives"] == 0


def test_missing_observability_never_claims_verified_research(tmp_path):
    campaign = make(tmp_path)
    snapshot = campaign.evaluator_snapshot()
    assert not snapshot["observability"]["verified"]
    assert snapshot["observability"]["protocol"] == "unverified_demo_presence_denominator"
    assert not snapshot["score"]["observability_verified"]


def test_crop_and_candidate_require_recorded_valid_boxes(tmp_path):
    campaign = make(tmp_path)
    campaign.act("launch")
    observation = campaign.act("capture")["observation"]
    box = {"x_min":250,"y_min":250,"x_max":750,"y_max":750}
    crop = campaign.act("inspect_crop",{"observation_id":observation["id"],"box":box})
    assert crop["ok"] and (tmp_path/crop["crop"]["image"]).exists()
    assert not campaign.act("record_candidate",{"evidence":[{"observation_id":"unknown","box":box}]})["ok"]
    assert campaign.act("record_candidate",{"evidence":[{"observation_id":observation["id"],"box":box}]})["ok"]
    assert not valid_box({**box,"x_max":2000})
    assert not valid_box({**box,"x_min":float("nan")})


def test_npz_physical_fields_and_private_sidecar(tmp_path):
    field_path = tmp_path/"physical.npz"
    temperature = np.ones((2,8,8))*285
    np.savez_compressed(field_path,times_s=[0.,86400.],x_m=np.linspace(0,64,8),y_m=np.linspace(0,64,8),
                        temperature_k=temperature,moisture=np.ones_like(temperature)*.2)
    field_path.with_suffix(".json").write_text(json.dumps({"fidelity":"experimental_coupled_finite_volume", "validation_state":"numerically_verified","research_accepted":False}))
    field_path.with_suffix(".truth.json").write_text(json.dumps({"targets":[{"id":"secret","x_m":32,"y_m":32,"radius_m":.3}]}))
    campaign = make(tmp_path/"images",thermal_field_path=str(field_path))
    campaign.act("launch")
    result = campaign.act("capture")
    assert result["ok"] and result["observation"]["fidelity"] == "experimental_coupled_finite_volume"
    assert not campaign.status()["research_accepted"]
    assert "secret" not in json.dumps(result)
    assert campaign.evaluator_snapshot()["observations"][0]["truth"][0]["observable"] is False


def test_npz_invalid_shape_rejected(tmp_path):
    path = tmp_path/"bad.npz"
    np.savez(path,times_s=[0,86400],x_m=[0,64],y_m=[0,64],temperature_k=np.zeros((3,2,2)),moisture=np.zeros((3,2,2)))
    with pytest.raises(ValueError,match="shape"):
        make(tmp_path,thermal_field_path=str(path))


def test_agent_adapter_only_returns_public_data(tmp_path):
    campaign = make(tmp_path)
    tools = AgentTools(campaign)
    result = tools.action("submit_report")
    assert result["submission"]["accepted"]
    assert "score" not in result
    assert "targets" not in tools.get_status()


def test_baseline_is_callable_without_oracle(tmp_path):
    campaign = make(tmp_path,camera={"width":128,"height":128})
    result = baseline_policy(campaign)
    assert result["ok"]
    assert result["status"]["completed"]
    assert result["status"]["sorties_used"] == 6
    assert len(result["status"]["observations"]) == 54


def test_arguments_and_events_remain_finite_json(tmp_path):
    campaign = make(tmp_path)
    assert not campaign.act("launch",[])["ok"]
    assert not campaign.act("launch",{"altitude_m":float("nan")})["ok"]
    json.dumps(campaign.evaluator_snapshot(),allow_nan=False)
    assert not valid_box({"x_min":"1","y_min":1,"x_max":100,"y_max":100})


def test_day_end_retains_landing_reserve(tmp_path):
    campaign = make(tmp_path,start_time_s=86350)
    result = campaign.act("launch")
    assert not result["ok"] and "landing" in result["error"]
    assert not campaign.airborne


def test_npz_cell_centres_cover_half_cell_boundary(tmp_path):
    path = tmp_path/"centres.npz"
    np.savez(path,times_s=[0,86400],x_m=[16,48],y_m=[16,48],temperature_k=np.ones((2,2,2))*285,moisture=np.ones((2,2,2))*.2)
    campaign = make(tmp_path,thermal_field_path=str(path))
    assert campaign.renderer.field is not None


def test_mcp_allowlist_and_actual_image_content(tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from aerorozum.mcp_server import create_server
    campaign = make(tmp_path)
    server = create_server(campaign)
    async def run():
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert "evaluator_snapshot" not in names and "reset" not in names
        await server.call_tool("launch",{})
        result = await server.call_tool("capture",{})
        content = result.content if hasattr(result,"content") else result[0]
        assert any(item.type == "image" for item in content)
        assert "private-" not in str([item.text for item in content if item.type=="text"])
    asyncio.run(run())


def test_spatiotemporal_coverage_requires_new_thermal_phase(tmp_path):
    campaign = make(tmp_path,demo_target_count=0,camera={"width":640,"height":512})
    campaign.act("launch");campaign.act("capture")
    morning = campaign.evaluator_snapshot()["coverage"]
    campaign.act("capture")
    assert campaign.evaluator_snapshot()["coverage"] == morning
    campaign.act("land")
    campaign.act("wait_until",{"time_s":40000})
    campaign.act("launch");campaign.act("capture")
    assert campaign.evaluator_snapshot()["coverage"] == 2*morning
    assert campaign.status()["coverage_by_phase"]["heating"] > 0
    assert campaign.status()["coverage_by_phase"]["cooling"] == 0


def test_public_arguments_cannot_override_internal_configuration(tmp_path):
    campaign = make(tmp_path)
    for arguments in ({"config":{"max_sorties":100}}, {"_targets":[]}, {"observability_verified":True}, {"thermal_field_path":"/etc/passwd"}):
        result = campaign.act("launch",arguments)
        assert not result["ok"]
        assert "immutable" in result["error"]
    assert campaign.config["max_sorties"] == 6 and not campaign.airborne
    assert not campaign.act("submit_report",{"reward":1})["ok"]
    assert not campaign.status()["research_accepted"]


def test_raw_observability_flag_does_not_enable_research_acceptance(tmp_path):
    campaign = make(tmp_path,observability_verified=True)
    assert not campaign.evaluator_snapshot()["observability"]["verified"]
    assert not campaign.status()["research_accepted"]
    with pytest.raises(ValueError,match="Research mode requires"):
        make(tmp_path,mode="research",observability_verified=True)


def test_certification_requires_accepted_physics(tmp_path):
    with pytest.raises(ValueError,match="accepted physical"):
        certify_observability({"camera":{"width":128,"height":128}},tmp_path/"manifest.json")


def test_physical_reference_certification_is_frozen_and_budget_checked(tmp_path):
    field_path = tmp_path/"accepted-test-fixture.npz"
    axis = np.linspace(0,64,64)
    xx,yy = np.meshgrid(axis,axis)
    temperature = 285+np.exp(-((xx-32)**2+(yy-32)**2)/(2*.8**2))
    np.savez_compressed(field_path,times_s=[0.,86400.],x_m=axis,y_m=axis,
                        temperature_k=np.stack([temperature,temperature]),moisture=np.ones((2,64,64))*.2)
    field_path.with_suffix(".json").write_text(json.dumps({"fidelity":"test_fixture_only","research_accepted":True,"validation_state":"test_fixture"}))
    field_path.with_suffix(".truth.json").write_text(json.dumps({"targets":[{"id":"private-fixture-target","x_m":32,"y_m":32,"radius_m":.8}]}))
    config = {"thermal_field_path":str(field_path),"camera":{"width":128,"height":128,"hfov_deg":20,"vfov_deg":20}}
    manifest_path = tmp_path/"certificate.json"
    manifest = certify_observability(config,manifest_path,seed=3)
    assert manifest["verified"]
    assert manifest["reference_sorties"] == 6 and manifest["reference_actions"] <= 128
    assert manifest["reference_airborne_seconds"] <= 3600
    assert manifest["observable_target_ids"] == ["private-fixture-target"]
    campaign = Campaign({**config,"observability_manifest_path":str(manifest_path)},seed=3,output_dir=tmp_path/"certified")
    assert campaign.status()["research_accepted"]
    assert campaign.evaluator_snapshot()["observability"]["verified"]
    assert "private-fixture-target" not in json.dumps(campaign.status())
    # A modified camera, seed, truth file or manifest cannot reuse a certificate.
    with pytest.raises(ValueError,match="configuration"):
        Campaign({**config,"observability_manifest_path":str(manifest_path)},seed=4,output_dir=tmp_path/"wrong-seed")
    manifest["observable_target_ids"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="integrity"):
        Campaign({**config,"observability_manifest_path":str(manifest_path)},seed=3,output_dir=tmp_path/"tampered")


def test_concurrent_captures_have_unique_authoritative_observations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    campaign = make(tmp_path)
    campaign.act("launch")
    before = campaign.time_s
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _:campaign.act("capture"),range(4)))
    assert all(r["ok"] for r in results)
    assert len({r["observation"]["id"] for r in results}) == 4
    assert campaign.time_s == before+8
    assert len(campaign.status()["observations"]) == 4
