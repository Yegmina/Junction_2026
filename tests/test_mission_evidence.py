import asyncio,copy,hashlib,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from aerorozum.mission_evidence import local_thermal_proposals,public_context,analyze_observation,record_observation_evidence,build_mission_report,score_mission_report
from aerorozum.labelling import MODEL

BOX={'x_min':200,'y_min':200,'x_max':400,'y_max':400}

def picture(tmp_path):
    path=tmp_path/'camera.png';image=Image.new('L',(128,128),100);draw=ImageDraw.Draw(image);draw.rectangle((28,32,39,43),fill=230);draw.rectangle((86,82,97,93),fill=10);image.save(path);return path

def context(obs='obs1'):
    return {'mission_id':'mission1','observation_id':obs,'source_kind':'gazebo_camera','sensor_modality':'simulated_thermal_intensity','sim_time_s':30,'pose':{'x_m':1,'y_m':2,'z_m':4},'capture':{'camera_topic':'/world/mission/model/drone/thermal','event_id':'frame1'}}


def test_reference_detects_both_hot_and_cold_without_claiming_mine_probability(tmp_path):
    image=picture(tmp_path);result=local_thermal_proposals(image)
    assert {'warmer','colder'}<={d['polarity'] for d in result['boxes']}
    assert result['model']=='local_thermal_contrast' and result['zero_candidates'] is None
    assert result['confidence_kind']=='uncalibrated_contrast_score' and not result['paid_request_issued']
    assert len(result['boxes'])<=12


def test_reference_ignores_flat_image_and_broad_regions(tmp_path):
    path=tmp_path/'flat.png';Image.new('L',(128,128),100).save(path)
    assert local_thermal_proposals(path)['boxes']==[]
    image=Image.open(path);ImageDraw.Draw(image).rectangle((20,20,100,100),fill=240);image.save(path)
    assert local_thermal_proposals(path)['boxes']==[]


def test_original_pixels_and_pose_are_bound_and_truth_is_removed(tmp_path):
    path=picture(tmp_path);before=path.read_bytes();ctx=context();ctx['truth']={'target_id':'SECRET_TARGET'};ctx['pose']['target']='SECRET_TARGET';ctx['camera']={'truth':'SECRET_TARGET'}
    result=record_observation_evidence(path,ctx,local_thermal_proposals(path),output_dir=tmp_path/'evidence')
    assert 'SECRET_TARGET' not in json.dumps(result)
    assert result['image_sha256']==hashlib.sha256(before).hexdigest() and path.read_bytes()==before
    assert result['overlay_geometry']=={'width':128,'height':128,'normalized_box_scale':1000,'resized':False}
    assert Image.open(tmp_path/'evidence'/result['overlay_image']).size==(128,128)
    changed=context('obs2');changed['pose']['x_m']=2
    other=record_observation_evidence(path,changed,local_thermal_proposals(path),output_dir=tmp_path/'evidence')
    assert other['evidence_id']!=result['evidence_id']


def test_replayed_reference_cannot_acquire_a_simulator_pose_or_wrong_capture_hash(tmp_path):
    ctx=context();ctx['source_kind']='recorded_reference'
    with pytest.raises(ValueError,match='moved pose'):public_context(ctx)
    image=picture(tmp_path);ctx=context();ctx['image_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA'):record_observation_evidence(image,ctx,local_thermal_proposals(image),output_dir=tmp_path/'evidence')


def test_high_bit_fields_need_an_explicit_display_mapping(tmp_path):
    path=tmp_path/'kelvin.png';Image.fromarray(np.full((32,32),30000,dtype=np.uint16)).save(path)
    with pytest.raises(ValueError,match='8-bit'):local_thermal_proposals(path)


class Provider:
    def __init__(self,output,rpm=10):
        self.output=output;self.calls=[];self.cfg={'reserve_output_tokens':2048,'thinking_budget':0,'api_version':'v1beta'};self.limiter=type('Limiter',(),{'rpm':rpm})();self.record=None
    def response_record(self,*args,**kwargs):return copy.deepcopy(self.record)
    async def generate(self,role,prompt,images,model):
        self.calls.append({'role':role,'prompt':prompt,'images':list(images),'model':model});await asyncio.sleep(.01)
        self.record={'model':model,'image_hashes':[hashlib.sha256(Path(images[0]).read_bytes()).hexdigest()],'output':copy.deepcopy(self.output),'raw_text':json.dumps(self.output)};return copy.deepcopy(self.output)


def lease(request):return {'allowed':True,'lease_id':'mission10rpm','allowance_rpm':10,'expires_at':(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),'global_quota_coordinated':True}


def test_no_implicit_paid_requests_and_no_empty_negative_on_missing_lease(tmp_path):
    image=picture(tmp_path);provider=Provider({'detections':[]})
    result=asyncio.run(analyze_observation(image,context(),provider=provider,output_dir=tmp_path/'out'))
    assert result['status']=='awaiting_request_lease' and result['zero_candidates'] is None and not provider.calls
    provider.limiter.rpm=120
    result=asyncio.run(analyze_observation(image,context(),provider=provider,request_authorizer=lease,output_dir=tmp_path/'out'))
    assert result['status']=='awaiting_request_lease' and not provider.calls


def test_exact_model_and_cache_reuse_preserve_evidence_and_never_send_context_truth(tmp_path):
    image=picture(tmp_path);provider=Provider({'detections':[{'bbox':BOX,'confidence':.8,'evidence':'compact signature'},{'bbox':{**BOX,'x_min':500,'x_max':600},'confidence':.2}]});ctx=context();ctx['truth']={'x':'SECRET_TARGET'}
    result=asyncio.run(analyze_observation(image,ctx,provider=provider,request_authorizer=lease,output_dir=tmp_path/'out'))
    assert result['status']=='analyzed' and len(result['boxes'])==len(result['uncertain_boxes'])==1
    assert len(provider.calls)==1 and provider.calls[0]['model']==MODEL and 'SECRET_TARGET' not in provider.calls[0]['prompt']
    cached=asyncio.run(analyze_observation(image,ctx,output_dir=tmp_path/'out'))
    assert cached['cache_source']=='mission_cache' and not cached['paid_request_issued'] and cached['evidence_id']==result['evidence_id']
    assert result['raw_response_retained'] and result['model']==MODEL


def test_same_pixels_concurrent_observations_make_one_request(tmp_path):
    image=picture(tmp_path);provider=Provider({'detections':[]})
    async def both():return await asyncio.gather(*(analyze_observation(image,context(f'obs{i}'),provider=provider,request_authorizer=lease,output_dir=tmp_path/'out') for i in range(2)))
    results=asyncio.run(both());assert len(provider.calls)==1 and all(x['zero_candidates'] is True for x in results)
    assert results[0]['evidence_id']!=results[1]['evidence_id']


def test_invalid_paid_output_is_retained_and_never_becomes_negative(tmp_path):
    image=picture(tmp_path);provider=Provider({})
    one=asyncio.run(analyze_observation(image,context(),provider=provider,request_authorizer=lease,output_dir=tmp_path/'out'))
    two=asyncio.run(analyze_observation(image,context(),provider=provider,request_authorizer=lease,output_dir=tmp_path/'out'))
    assert one['status']==two['status']=='invalid_model_response' and one['zero_candidates'] is None
    assert len(provider.calls)==1 and len(list((tmp_path/'out/private/paid').glob('*.json')))==1


def score_fixture(tmp_path):
    image=picture(tmp_path);record=record_observation_evidence(image,context(),{'model':'local_thermal_contrast','provenance':'pixel_heuristic','boxes':[{'bbox':BOX,'confidence':.8}]},output_dir=tmp_path/'out')
    track={'candidate_id':'c1','evidence':[{'evidence_id':record['evidence_id'],'bbox':BOX}]};report=build_mission_report('mission1',[record],candidate_tracks=[track]);reference={'reference_kind':'simulator_private','reference_scope':'complete','frozen_before_run':True,'eligible_target_ids':['a','unvisited'],'observability_verified':False,'observations':{'obs1':{'image_sha256':record['image_sha256'],'targets':[{'target_id':'a','bbox':BOX,'observable':True}]}}};return record,report,reference


def test_privileged_scoring_penalizes_unvisited_targets_and_duplicate_predictions(tmp_path):
    record,report,reference=score_fixture(tmp_path);score=score_mission_report(report,[record],reference,qualified_coverage=.5)
    assert score['true_positives']==1 and score['false_negatives']==1 and score['recall']==.5
    assert score['negative_scene_false_alarm_rate'] is None and not score['independent_real_mine_accuracy_claim']
    duplicate=copy.deepcopy(report['candidate_tracks'][0]);duplicate['candidate_id']='c2';report['candidate_tracks'].append(duplicate)
    score=score_mission_report(report,[record],reference,qualified_coverage=.5)
    assert score['true_positives']==score['false_positives']==1


def test_scoring_rejects_teacher_as_independent_truth_and_mismatched_images(tmp_path):
    record,report,reference=score_fixture(tmp_path);reference['reference_kind']='teacher_machine'
    with pytest.raises(ValueError,match='independent'):score_mission_report(report,[record],reference,qualified_coverage=.5)
    reference['reference_kind']='simulator_private';reference['observations']['obs1']['image_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):score_mission_report(report,[record],reference,qualified_coverage=.5)


def test_unreviewed_recorded_background_prevents_closed_world_false_positive_claims(tmp_path):
    record,report,reference=score_fixture(tmp_path);reference['reference_scope']='inserted_objects_only'
    score=score_mission_report(report,[record],reference,qualified_coverage=.5)
    assert score['recall']==.5 and score['false_positives'] is None and score['f2'] is None and score['reward'] is None


def test_report_rejects_unbound_evidence_and_does_not_invent_unique_mines(tmp_path):
    record,report,_=score_fixture(tmp_path)
    assert report['unique_physical_mines_confirmed']==0 and report['counts']['candidate_box_measurements']==1
    with pytest.raises(ValueError,match='missing'):build_mission_report('mission1',[record],candidate_tracks=[{'candidate_id':'bad','evidence':[{'evidence_id':'absent','bbox':BOX}]}])


def test_atomic_capture_adapter_uses_receipt_pose_and_refuses_unbound_metadata(tmp_path):
    from aerorozum.mission_evidence import context_from_capture
    image=picture(tmp_path);receipt={'atomic_capture':True,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'pose':{'x_m':12,'y_m':3,'z_m':5,'yaw_rad':.2},'pose_frame':'drone_base_in_world','sim_time_s':44.2,'camera_topic':'/native/thermal','event_id':'native123'}
    ctx=context_from_capture(image,receipt,mission_id='mission1',observation_id='obs1')
    assert ctx['pose']['x_m']==12 and ctx['sim_time_s']==44.2
    assert public_context(ctx)['capture']['binding']=='atomic_image_pose_sha256_verified'
    receipt['atomic_capture']=False
    with pytest.raises(ValueError,match='Atomic'):context_from_capture(image,receipt,mission_id='mission1',observation_id='obs1')
    receipt['atomic_capture']=True;receipt['image_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA'):context_from_capture(image,receipt,mission_id='mission1',observation_id='obs1')
