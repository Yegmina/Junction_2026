import hashlib,json
from pathlib import Path
from aerorozum.flight_filter import read_telemetry,classify_frame,filter_rows


def telemetry(tmp_path,events):
    path=tmp_path/'telemetry.jsonl';path.write_text(''.join(json.dumps({'received_at_ns':int(t*1e9),'system':2,'component':1,'message':message})+'\n' for t,message in events));return read_telemetry(path,{'object':'jetson/session/telemetry.jsonl','generation':'100'})


def frame(time=101,**kwargs):return {'id':'f','sha256':'a'*64,'timestamp_s':time,'time_basis':'jetson_received_at_ns','source':{'session_id':'s'},**kwargs}


def test_only_fresh_airborne_primary_passes(tmp_path):
    data=telemetry(tmp_path,[(100,{'mavpackettype':'EXTENDED_SYS_STATE','landed_state':2}),(100.5,{'mavpackettype':'ALTITUDE','altitude_relative':8.})])
    result=classify_frame(frame(),data)
    assert result['status']=='eligible' and result['airborne_eligible']
    assert result['relative_altitude_m']==8 and result['landed_state']==2 and result['state_age_s']==1
    assert result['telemetry']['generation']=='100'
    assert classify_frame(frame(103),data)['status']=='unknown'


def test_armed_and_high_altitude_alone_never_prove_airborne(tmp_path):
    data=telemetry(tmp_path,[(100,{'mavpackettype':'HEARTBEAT','base_mode':128}),(100,{'mavpackettype':'ALTITUDE','altitude_relative':10})])
    assert classify_frame(frame(),data)['reason']=='no_fresh_primary_landed_state'


def test_on_ground_and_transitional_states_are_excluded(tmp_path):
    for state,expected in [(1,'ground'),(0,'unknown'),(3,'unknown'),(4,'unknown')]:
        data=telemetry(tmp_path,[(100,{'mavpackettype':'EXTENDED_SYS_STATE','landed_state':state}),(100,{'mavpackettype':'ALTITUDE','altitude_relative':10})])
        assert classify_frame(frame(),data)['status']==expected


def test_video_time_only_and_missing_telemetry_fail_closed(tmp_path):
    assert classify_frame(frame(time_basis='container_pts_seconds',timestamp_s=None),{})['reason']=='thermal_wallclock_alignment_unavailable'
    assert classify_frame(frame(),None)['reason']=='telemetry_unavailable'
    assert classify_frame(frame(receipt_timestamp_ns=101_000_000_000,wallclock_alignment_available=False),{})['status']=='unknown'


def test_home_relative_altitude_is_diagnostic_and_never_overrides_primary(tmp_path):
    data=telemetry(tmp_path,[(100,{'mavpackettype':'EXTENDED_SYS_STATE','landed_state':2}),(102,{'mavpackettype':'ALTITUDE','altitude_relative':8})])
    assert classify_frame(frame(),data)['status']=='eligible'
    assert classify_frame(frame(),data)['altitude_diagnostic_status']=='unavailable'
    data=telemetry(tmp_path,[(100,{'mavpackettype':'EXTENDED_SYS_STATE','landed_state':2}),(100,{'mavpackettype':'GLOBAL_POSITION_INT','relative_alt':-16000})])
    assert classify_frame(frame(),data)['status']=='eligible'
    assert classify_frame(frame(),data)['relative_altitude_m']==-16
    assert classify_frame(frame(),data)['relative_altitude_reference']=='HOME, not height above ground'


def test_portable_manifest_hash_binds_all_skip_decisions(tmp_path):
    data=telemetry(tmp_path,[(100,{'mavpackettype':'EXTENDED_SYS_STATE','landed_state':2}),(100,{'mavpackettype':'ALTITUDE','altitude_relative':3})])
    audit,manifest=filter_rows([frame(),frame(id='unknown',time_basis='container_pts_seconds')],{'s':data},session_splits={'s':'train'},output_dir=tmp_path/'out')
    assert audit['counts']=={'total_frames':2,'airborne_eligible':1,'ground_excluded':0,'unknown_skipped':1}
    assert manifest['status']=='airborne_only' and len(manifest['images'])==1
    assert manifest['images'][0]['eligibility']['status']=='eligible' and manifest['images'][0]['airborne_eligible']
    assert hashlib.sha256((tmp_path/'out/airborne-eligibility.json').read_bytes()).hexdigest()==manifest['eligibility_manifest_sha256']
    assert audit['paid_inference_issued'] is False
