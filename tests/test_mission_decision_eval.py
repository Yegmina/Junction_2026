from cloud.mission_decision_eval import score
import json

def test_detection_only_output_cannot_pass_as_agent_action():
    c=dict(images=[],public_state=dict(allowed_areas=['east']),expected=dict(action='inspect_area',allowed_area_ids=['east'],max_altitude_m=10))
    s=score(c,'{"detections":[]}');assert s['json_valid'] and not s['action_valid'] and not s['contract_success']

def test_forbidden_region_fails_even_with_valid_json():
    c=dict(images=[{}],public_state=dict(allowed_areas=['north']),expected=dict(action='inspect_area',allowed_area_ids=['north'],max_altitude_m=10))
    s=score(c,json.dumps(dict(action='inspect_area',area_id='west',altitude_m=3)));assert not s['action_valid'] and not s['contract_success']

def test_inventing_detections_fails_no_evidence_contract():
    c=dict(images=[],public_state=dict(allowed_areas=['south']),expected=dict(action='inspect_area',allowed_area_ids=['south'],max_altitude_m=10,no_detections=True))
    s=score(c,json.dumps(dict(action='inspect_area',area_id='south',altitude_m=3,detections=[{'bbox':{}}])));assert s['hallucinated_without_image'] and not s['contract_success']

def test_always_return_does_not_win_inspection_case():
    c=dict(images=[],public_state=dict(allowed_areas=['east']),expected=dict(action='inspect_area',allowed_area_ids=['east'],max_altitude_m=10))
    assert not score(c,'{"action":"return_home"}')['contract_success']
