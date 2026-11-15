import hashlib
import json
from pathlib import Path
import pytest
from aerorozum.training import checked_box,detection_score,detection_reward,load_training_manifest,language_lora_targets,policy_conversations

B={'x_min':100,'y_min':100,'x_max':200,'y_max':200}
C={'x_min':400,'y_min':400,'x_max':500,'y_max':500}

def manifest(tmp_path,rows):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'metadata':{'status':'passed'},'images':rows}));return path

def row(tmp_path,**changes):
    image=tmp_path/'frame.png';image.write_bytes(b'image-bytes')
    value={'id':'one','image':'frame.png','sha256':hashlib.sha256(b'image-bytes').hexdigest(),'split':'train','zone':'zone1','provenance':'original_manual','boxes':[{'label':'mine','bbox':B}]}
    value.update(changes);return value

def test_rewards_correct_boxes_and_penalizes_false_positives():
    perfect=json.dumps({'detections':[{'bbox':B}]})
    duplicate=json.dumps({'detections':[{'bbox':B},{'bbox':B}]})
    wrong=json.dumps({'detections':[{'bbox':C}]})
    rewards=detection_reward([perfect,duplicate,wrong,'not-json'],solution=[[B],[B],[B],[B]])
    assert rewards[0]==1 and 0<rewards[1]<1 and rewards[2]==0 and rewards[3]==-1

def test_negative_image_requires_empty_predictions():
    assert detection_reward(['{"detections":[]}'],solution=[[]])==[1]
    assert detection_reward([json.dumps({'detections':[{'bbox':B}]})],solution=[[]])==[0]

def test_continuous_stream_reward_resolves_localization_without_box_spam():
    from aerorozum.training import dense_detection_reward
    shifted={**B,'x_min':110,'x_max':210};full={'x_min':0,'y_min':0,'x_max':1000,'y_max':1000}
    outputs=[json.dumps({'detections':[{'bbox':b} for b in boxes]}) for boxes in ([B],[shifted],[B,B],[full],[])]
    perfect,near,duplicate,oversized,empty=dense_detection_reward(outputs,solution=[[B]]*5)
    assert perfect==1 and 0<near<1 and 0<duplicate<1 and oversized<near and empty==0
    assert dense_detection_reward([outputs[0],outputs[-1],'broken'],solution=[[],[],[B]])==[0.,1.,-1.]

def test_box_spam_and_oversized_box_do_not_increase_reward():
    full={'x_min':0,'y_min':0,'x_max':1000,'y_max':1000}
    assert detection_score([full],[B])['tp']==0
    assert detection_score([B,B,B],[B])['fp']==2
    assert detection_score([B,C],[B,C])['f2']==1

def test_nonfinite_and_inverted_boxes_rejected():
    for changes in ({'x_min':float('nan')},{'x_max':-1},{'y_max':1001}):
        with pytest.raises(ValueError):checked_box({**B,**changes})

def test_compatibility_selects_train_and_preserves_provenance(tmp_path):
    rows=[row(tmp_path),row(tmp_path,id='held',sha256=None,split='test',zone='zone8')]
    selected,info=load_training_manifest(manifest(tmp_path,rows),'compatibility',tmp_path)
    assert len(selected)==1 and info['research_result'] is False
    assert info['provenance']=={'original_manual':1}

def test_cross_split_duplicate_is_rejected(tmp_path):
    train=row(tmp_path)
    with pytest.raises(ValueError,match='crosses'):
        load_training_manifest(manifest(tmp_path,[train,{**train,'id':'different-name','split':'test','zone':'zone8'}]),'compatibility',tmp_path)

def test_zone_leak_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='zone leaked'):
        load_training_manifest(manifest(tmp_path,[row(tmp_path,zone='zone8')]),'compatibility',tmp_path)

def test_real_only_cannot_be_called_research_training(tmp_path):
    with pytest.raises(ValueError,match='accepted synthetic'):
        load_training_manifest(manifest(tmp_path,[row(tmp_path)]),'sft',tmp_path)

def test_accepted_synthetic_and_reviewed_real_can_train(tmp_path):
    selected,info=load_training_manifest(manifest(tmp_path,[row(tmp_path,provenance='direct_physics',research_accepted=True,fidelity='direct')]),'sft',tmp_path)
    assert len(selected)==1 and info['synthetic_images']==1 and info['research_result'] is True

def test_unaccepted_synthetic_and_demo_are_rejected(tmp_path):
    for changes in ({'provenance':'direct_physics'},{'fidelity':'demonstration_analytic_not_validated'}):
        with pytest.raises(ValueError):load_training_manifest(manifest(tmp_path,[row(tmp_path,**changes)]),'compatibility',tmp_path)

def test_artifact_escape_and_checksum_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='escapes'):
        load_training_manifest(manifest(tmp_path,[row(tmp_path,image='../secret.png')]),'compatibility',tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        load_training_manifest(manifest(tmp_path,[row(tmp_path,sha256='wrong')]),'compatibility',tmp_path)

def test_lora_only_targets_language_attention():
    class Model:
        def named_modules(self):return [(name,None) for name in ['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj','model.vision_model.layers.0.self_attn.q_proj','model.audio_model.layers.0.self_attn.q_proj','model.multi_modal_projector.q_proj']]
    targets=language_lora_targets(Model())
    assert len(targets)==2 and all('language_model' in x for x in targets)


def test_visual_projection_scope_is_explicit_and_does_not_unfreeze_audio():
    from aerorozum.training import perception_lora_targets
    from types import SimpleNamespace
    class Model:
        def named_modules(self):return [(name,SimpleNamespace(in_features=3840,out_features=3840)) for name in [
            'model.language_model.layers.0.self_attn.q_proj',
            'model.embed_vision.multimodal_embedder.embedding_projection','model.embed_audio.embedding_projection']]
    assert perception_lora_targets(Model())==['model.language_model.layers.0.self_attn.q_proj']
    assert perception_lora_targets(Model(),'language_visual_projection')==[
        'model.language_model.layers.0.self_attn.q_proj','model.embed_vision.multimodal_embedder.embedding_projection']
    with pytest.raises(ValueError,match='Unknown'):perception_lora_targets(Model(),'all')
    class Missing(Model):
        def named_modules(self):return super().named_modules()[:1]
    with pytest.raises(ValueError,match='Expected one'):perception_lora_targets(Missing(),'language_visual_projection')

def test_policy_adapter_rejects_hidden_truth():
    with pytest.raises(ValueError,match='Hidden state'):
        policy_conversations([{'split':'train','research_accepted':True,'truth':{},'messages':[],'tools':[{}]}])


def test_policy_window_replays_public_prefix_and_keeps_rewards_private(tmp_path):
    from aerorozum.training import SurveyPolicyEnv
    state={'campaign_config':{},'campaign_id':'abcdef123456abcdef123456','seed':17,'prefix_actions':[{'action':'launch','arguments':{'altitude_m':4}}]}
    env=SurveyPolicyEnv(tmp_path,allow_demo=True)
    env.reset(json.dumps(state))
    assert env.get_reward()==0
    blocks=env.act('capture','{}')
    visible=json.loads(blocks[0]['text'])
    assert visible['ok'] and visible['observation']['width']==640
    assert blocks[1]['type']=='image'
    assert 'truth' not in visible and 'score' not in visible
    assert 'targets' not in visible['status']
    assert isinstance(env.get_reward(),float)
    for _ in range(8):env.act('inspect_observation',json.dumps({'observation_id':visible['observation']['id']}))
    assert 'exhausted' in env.act('capture','{}')[0]['text']


def test_policy_grpo_rejects_demo_even_if_record_claims_acceptance(tmp_path):
    from aerorozum.training import SurveyPolicyEnv
    state={'campaign_config':{},'campaign_id':'abcdef123456abcdef123456','seed':17,'prefix_actions':[]}
    with pytest.raises(ValueError,match='accepted physical'):
        SurveyPolicyEnv(tmp_path).reset(json.dumps(state))


def test_policy_split_leak_and_nonphysical_training_rejected(tmp_path):
    from aerorozum.training import load_policy_manifest
    row={'id':'x','run_id':'one','split':'train','provenance':'demo_gemini_policy_trajectory','research_accepted':False,'campaign_config':{},'campaign_id':'abcdef123456abcdef123456','prefix_actions':[],'seed':17,'prompt':'Inspect the public scene','completion':json.dumps({'action':'capture','arguments':{}}),'images':[]}
    p=tmp_path/'policy.jsonl';p.write_text(json.dumps(row)+'\n')
    rows,meta=load_policy_manifest(p,'policy_compatibility',tmp_path)
    assert len(rows)==1 and not meta['research_result']
    with pytest.raises(ValueError):load_policy_manifest(p,'policy_sft',tmp_path)
    p.write_text(json.dumps(row)+'\n'+json.dumps({**row,'split':'test'})+'\n')
    with pytest.raises(ValueError,match='crosses'):load_policy_manifest(p,'policy_compatibility',tmp_path)


def real_manifest(tmp_path,positive=32,negative=32):
    rows=[]
    for i in range(positive+negative):
        payload=f'unique real frame {i}'.encode();name=f'real-{i}.jpg';(tmp_path/name).write_bytes(payload)
        is_positive=i<positive;detections=[{'bbox':B,'confidence':.8}] if is_positive else []
        rows.append({'id':str(i),'image':name,'sha256':hashlib.sha256(payload).hexdigest(),'split':'train',
            'source':{'session_id':'session-a' if i%2==0 else 'session-b','video_sha256':('a' if i%2==0 else 'b')*64},
            'model':'gemini-robotics-er-2-preview','provenance':'user_accepted_model','user_threshold':.6,'human_reviewed':False,
            'detections':detections,'model_output':{'detections':detections},'boxes':[{'bbox':B,'confidence':.8}] if is_positive else [],'uncertain_boxes':[],'zero_candidates':not is_positive})
    path=tmp_path/'real-transfer.json';path.write_text(json.dumps({'metadata':{'status':'passed'},'session_splits':{'session-a':'train','session-b':'train'},'images':rows}))
    return path,rows


def test_real_transfer_accepts_authorized_labels_and_balances_boot_negatives(tmp_path):
    from aerorozum.training import load_real_transfer_manifest
    path,rows=real_manifest(tmp_path,positive=32,negative=200)
    selected,protocol=load_real_transfer_manifest(path,'real_sft',root=tmp_path,seed=17)
    assert len(selected)==96 and protocol['positive_frames']==32 and protocol['zero_candidate_frames']==64
    assert protocol['experiment_namespace']=='real-transfer-machine-labels-v1'
    assert protocol['human_reviewed'] is False and protocol['physics_acceptance_claimed'] is False
    assert any(int(r['id'])>100 for r in selected if r['zero_candidates'])
    again,_=load_real_transfer_manifest(path,'real_sft',root=tmp_path,seed=17)
    assert [r['id'] for r in selected]==[r['id'] for r in again]


def test_real_transfer_needs_both_classes(tmp_path):
    from aerorozum.training import load_real_transfer_manifest
    path,rows=real_manifest(tmp_path,positive=0,negative=100)
    with pytest.raises(ValueError,match='32 unique positive'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)


def test_real_transfer_excludes_mixed_uncertainty(tmp_path):
    from aerorozum.training import load_real_transfer_manifest
    path,rows=real_manifest(tmp_path)
    rows[0]['uncertain_boxes']=[{'bbox':C,'confidence':.4}]
    d=json.loads(path.read_text());d['images']=rows;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='31 positive'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)


def test_real_transfer_enforces_frozen_session_and_exact_labeler(tmp_path):
    from aerorozum.training import load_real_transfer_manifest
    path,rows=real_manifest(tmp_path)
    d=json.loads(path.read_text());d['images'][0]['split']='test';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='frozen recording-session'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)
    d['images'][0]['split']='train';d['images'][0]['model']='another-model';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='exact Robotics'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)


def test_real_transfer_verifies_frame_checksum_and_zero_response(tmp_path):
    from aerorozum.training import load_real_transfer_manifest
    path,rows=real_manifest(tmp_path)
    (tmp_path/rows[0]['image']).write_bytes(b'changed pixels')
    with pytest.raises(ValueError,match='checksum mismatch'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)


def test_real_grpo_requires_completed_corresponding_frozen_sft(tmp_path):
    from aerorozum.training import validate_parent_adapter
    parent=tmp_path/'real_sft'/'seed-17'/'adapter';parent.mkdir(parents=True)
    (parent/'adapter_config.json').write_text('{}')
    with pytest.raises(ValueError,match='completion summary'):
        validate_parent_adapter(parent,'real_sft','google/gemma-4-12B-it',17,'frozen')
    summary={'status':'completed','stage':'real_sft','model':'google/gemma-4-12B-it','seed':17,'manifest_hash':'frozen'}
    (parent.parent/'summary.json').write_text(json.dumps(summary))
    assert validate_parent_adapter(parent,'real_sft','google/gemma-4-12B-it',17,'frozen')['status']=='completed'
    with pytest.raises(ValueError,match='does not match'):
        validate_parent_adapter(parent,'real_sft','google/gemma-4-12B-it',43,'frozen')


def test_real_transfer_archive_provenance_requires_original_member_hash(tmp_path,monkeypatch):
    from aerorozum.training import load_real_transfer_manifest
    from zipfile import ZipFile
    path,rows=real_manifest(tmp_path)
    archive=tmp_path/'source.zip'
    with ZipFile(archive,'w') as z:
        for i,row in enumerate(rows):z.writestr(f'data/03032020__zone1__frame_{i}.jpg',(tmp_path/row['image']).read_bytes())
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    for i,row in enumerate(rows):row['source']={'source_archive_sha256':archive_hash,'parent_image_sha256':row['sha256'],'archive_member':f'data/03032020__zone1__frame_{i}.jpg','zone':'zone1'}
    path.write_text(json.dumps({'metadata':{'status':'passed'},'images':rows}));monkeypatch.setenv('AEROROZUM_SOURCE_ARCHIVE',str(archive))
    selected,protocol=load_real_transfer_manifest(path,'real_sft',root=tmp_path)
    assert len(selected)==64 and all(r['source_group'].startswith('archive:') for r in selected)
    rows[0]['source']['parent_image_sha256']='0'*64;path.write_text(json.dumps({'metadata':{},'images':rows}))
    with pytest.raises(ValueError,match='Parent image checksum'):
        load_real_transfer_manifest(path,'real_sft',root=tmp_path)


def test_freeze_real_transfer_combines_cached_sources_without_changing_main_protocol(tmp_path):
    from aerorozum.training import freeze_real_transfer_manifest,load_real_transfer_manifest
    source,rows=real_manifest(tmp_path,positive=32,negative=80)
    positive=tmp_path/'positive-labels.json';negative=tmp_path/'negative-labels.json'
    positive.write_text(json.dumps({'images':rows[:32]}));negative.write_text(json.dumps({'images':rows[32:]}))
    frozen=tmp_path/'frozen-transfer.json'
    result=freeze_real_transfer_manifest([positive,negative],frozen,{'sessions':{'session-a':'train','session-b':'train'}},root=tmp_path)
    assert result['status']=='frozen' and result['training_images']==96
    assert json.loads(source.read_text())['images']==rows
    again=freeze_real_transfer_manifest([positive,negative],frozen,{'sessions':{'session-a':'train','session-b':'train'}},root=tmp_path)
    assert again['status']=='already_frozen' and again['manifest_hash']==result['manifest_hash']
    assert load_real_transfer_manifest(frozen,'real_grpo',root=tmp_path)[1]['manifest_hash']==result['manifest_hash']
