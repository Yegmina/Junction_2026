import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from aerorozum.common import save,digest
from aerorozum.fast_fields import file_hash,validate_pair,render_pair_shard,SurfaceField
from aerorozum.streaming import StreamCoordinator
from aerorozum.training import load_stream_manifest


def accepted_pair(root):
    axis=np.linspace(.025,.975,20);xx,yy=np.meshgrid(axis,axis)
    paths={}
    for name,times in (('present',[0.,43200.,86400.]),('absent',[0.,21600.,86400.])):
        values=np.ones((len(times),20,20))*285
        if name=='present':values+=np.exp(-((xx-.5)**2+(yy-.5)**2)/(2*.1**2))[None,:,:]
        path=root/(name+'.npz');np.savez_compressed(path,times_s=times,x_m=axis,y_m=axis,temperature_k=values,moisture=np.ones_like(values)*.2)
        save(path.with_suffix('.json'),{'backend':'dumux_3.10.0','research_accepted':True,'validation_state':'numerically_accepted','field_sha256':file_hash(path),'domain':{'width_m':1,'height_m':1}})
        save(path.with_suffix('.truth.json'),{'targets':[]});paths[name]=path
    certificate=root/'pair.json';save(certificate,{'accepted':True,'status':'numerically_accepted_pair','research_accepted':True,
        'present_sha256':file_hash(paths['present']),'absent_sha256':file_hash(paths['absent'])})
    return {'campaign_id':'unit-fixture-only','split':'train','status':'accepted','field_path':str(paths['present']),
        'counterfactual_path':str(paths['absent']),'acceptance_report_path':str(certificate)}


def test_individual_pair_immediately_renders_mmap_shard_without_other183(tmp_path):
    entry=accepted_pair(tmp_path)
    doc=render_pair_shard(entry,tmp_path/'pipeline/synthetic',tmp_path,count=8,shape=(128,128))
    assert len(doc['images'])==8 and any(row['boxes'] for row in doc['images'])
    assert all(row['field_domain_m']==[1.,1.] and row['implied_nadir_altitude_m']==4 for row in doc['images'])
    assert all(row['width']<20 and row['height']<20 and row['effective_crop_fov_deg'][0]<20 for row in doc['images'])
    assert all(row['provenance']=='direct_physics' and row['acceptance_certificate_sha256'] for row in doc['images'])
    assert all(row['training_role']=='auxiliary_physics_cue_only' and row['camera_transfer']['gaussian_noise_std_k']==.04 for row in doc['images'])
    assert not any((tmp_path/'pipeline/synthetic/.field-cache').glob('*'))
    field=SurfaceField(entry['field_path'],tmp_path/'cache')
    assert isinstance(field.data['temperature_k'],np.memmap)
    with pytest.raises(ValueError,match='extrapolation'):field.sample(np.array([.5]),np.array([.5]),90000)
    again=render_pair_shard(entry,tmp_path/'pipeline/synthetic',tmp_path,count=8,shape=(128,128))
    assert again==doc


def test_unaccepted_or_modified_pair_cannot_render_training_images(tmp_path):
    entry=accepted_pair(tmp_path);entry['status']='experimental'
    with pytest.raises(ValueError):render_pair_shard(entry,tmp_path/'shards',tmp_path,count=8)
    entry['status']='accepted';Path(entry['field_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):validate_pair(entry)

def test_explicit_physical_split_required_and_field_lineage_cannot_cross_holdout(tmp_path):
    entry=accepted_pair(tmp_path)
    with pytest.raises(ValueError,match='split'):validate_pair({k:v for k,v in entry.items() if k!='split'})
    worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());manifests=[];documents=[]
    for split,seed in (('train',17),('validation',43)):
        doc=render_pair_shard({**entry,'campaign_id':'unit-fixture-'+split,'split':split},tmp_path/'pipeline/synthetic',tmp_path,count=8,seed=seed)
        path=(tmp_path/doc['images'][0]['image']).parents[1]/'manifest.json';documents.append(doc)
        worker.state['accepted_pairs'][split]={'manifest':str(path),'acceptance_report_path':entry['acceptance_report_path']}
    assert {r['sha256'] for r in documents[0]['images']}.isdisjoint({r['sha256'] for r in documents[1]['images']})
    rows,*_=worker._catalog()
    assert rows and not any(r['split']=='train' for r in rows)

def test_bridge_redacts_test_labels_and_paths_but_keeps_hash_exclusion(tmp_path):
    from aerorozum.pipeline_bridge import redacted_bridge_snapshot
    doc=real_index(tmp_path);held=copy.deepcopy(doc['images'][24]);held.update(id='sealed-private-label-id',split='test',image='sealed/zone9-private.jpg')
    doc['images'].append(held);doc['source_assets']=[{'image':'sealed/zone9-private.jpg','boxes':['sealed-coordinates']}]
    exported=redacted_bridge_snapshot(doc)
    assert 'sealed-private-label-id' not in json.dumps(exported) and 'zone9-private' not in json.dumps(exported)
    assert held['sha256'] in exported['sealed_test_image_hashes'] and len(doc['images'])==33
    save(tmp_path/'pipeline/inputs/real.json',exported)
    worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());rows,*_=worker._catalog()
    assert not any(r['sha256']==held['sha256'] for r in rows)


def real_index(root):
    rows=[];box={'bbox':{'x_min':100,'y_min':100,'x_max':400,'y_max':400}}
    for i in range(32):
        path=root/'images'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(32,32),(i*7%255,i*13%255,i*19%255)).save(path)
        manual=i<24;validation=16<=i<24;positive=i<28
        row={'id':str(i),'image':str(path.relative_to(root)),'sha256':file_hash(path),'split':'validation' if validation else 'train',
             'training_eligible':not validation,'cross_split_duplicate':False,
             'boxes':[box] if positive else [],'source_group':'manual-zone7' if validation else 'manual-zone1' if manual else 'native-one',
             'provenance':'original_manual' if manual else 'user_accepted_model'}
        if manual:row['zone']='zone7' if validation else 'zone1'
        else:row.update(model='gemini-robotics-er-2-preview',user_threshold=.6,human_reviewed=False,
            source={'session_id':'native-one','video_sha256':'a'*64},detections=[{**box,'confidence':.9}] if positive else [],zero_candidates=not positive,uncertain_boxes=[])
        rows.append(row)
    doc={'images':rows,'session_splits':{'native-one':'train'},'counts':{'images':len(rows)}}
    save(root/'pipeline/inputs/real.json',doc);return doc

class Process:
    pid=os.getpid()
    def poll(self):return None

def settings():return {'minimum_native_positive':2,'minimum_native_zero':2,'minimum_new_rows':4,
    'maximum_new_rows':8,'manual_anchors':16,'replay_rows':4,'deadline_utc':'2099-01-01T00:00:00+00:00'}

def test_native_arrival_freezes_immutable_anchored_batch_and_starts_two_gpu_groups(tmp_path):
    calls=[];worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:(calls.append(args) or Process()))
    before=worker.tick();assert before['stats']['training_batches']==0
    real_index(tmp_path);after=worker.tick()
    assert after['stats']['training_batches']==1 and len(calls)==2
    assert not ({r['id'] for r in after['media']} & {str(i) for i in range(16,24)})
    assert all(r['batch_id']==worker.state['batches'][0]['id'] for r in after['media'])
    assert [job['seed'] for job in worker.state['jobs']]==[17,43,91]
    assert {tuple(job['gpu_group']) for job in worker.state['jobs'] if job['status']=='running'}=={(0,1),(2,3)}
    batch=Path(worker.state['batches'][0]['manifest']);original=batch.read_bytes()
    rows,protocol=load_stream_manifest(batch,'stream_sft',root=tmp_path)
    assert len(rows)==24 and protocol['provenance']['original_manual']==16
    worker.tick();assert batch.read_bytes()==original and len(calls)==2
    assert all(row['split']=='train' and row.get('zone')!='zone7' for row in rows)

def test_restart_resumes_queue_without_duplicating_batches(tmp_path):
    real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    restored=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());restored.tick()
    assert len(restored.state['batches'])==1 and len(restored.state['jobs'])==3

def test_heldout_and_uncertain_native_never_enter_training(tmp_path):
    document=real_index(tmp_path)
    for row in document['images']:
        if row['provenance']=='user_accepted_model':row.update(split='test')
    document['session_splits']['native-one']='test';save(tmp_path/'pipeline/inputs/real.json',document)
    worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    assert not worker.state['batches']

def test_completed_text_without_actual_adapters_cannot_finish_a_job(tmp_path):
    real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    job=worker.state['jobs'][0]
    save(Path(job['output'])/'result.json',{'status':'completed','batch_id':job['batch_id'],'seed':job['seed'],'promoted':True})
    worker.manage_jobs();assert job['status']=='running' and not worker.state['promoted']

def test_heldout_only_index_growth_does_not_queue_duplicate_train_passes(tmp_path):
    doc=real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    frozen=Path(worker.state['batches'][0]['manifest']).read_bytes()
    for row in doc['images']:
        if row['split']=='validation':row['new_validation_note']='arrived later'
    doc['counts']['images']+=1000;save(tmp_path/'pipeline/inputs/real.json',doc);worker.tick()
    assert len(worker.state['batches'])==1 and len(worker.state['jobs'])==3
    assert Path(worker.state['batches'][0]['manifest']).read_bytes()==frozen

def test_single_gpu_probe_waits_for_a_real_free_group(tmp_path):
    calls=[];real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:(calls.append(args) or Process()));worker.tick()
    worker.manage_probe({'request_single_gpu_probe':True},{0,1,2,3})
    assert 'gpu_probe' not in worker.state
    worker.manage_probe({'request_single_gpu_probe':True},{2,3})
    assert worker.state['gpu_probe']['reserved_group']==[0,1]
    assert calls[-1][1]['CUDA_VISIBLE_DEVICES']=='0'
    assert 'measured_topology' not in worker.state

def test_single_gpu_topology_requires_successful_measured_backward_and_grpo(tmp_path):
    real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    output=tmp_path/'pipeline/probe';output.mkdir()
    worker.state['gpu_probe']={'status':'running','pid':999999999,'reserved_group':[0,1],'output':str(output)}
    save(output/'result.json',{'status':'passed','single_gpu':True,'selected':{'memory_fraction':.85,'forward_backward_verified':True,'grpo_verified':True,'microbatch':2,'gradient_accumulation':2}})
    worker.manage_probe({},set())
    assert worker.state['measured_topology']['gpu_groups']==[[0],[1],[2],[3]]
    assert worker.state['measured_topology']['microbatch']==2

def test_numerical_patch_arrival_is_auxiliary_and_never_replaces_recorded_appearance(tmp_path):
    real_index(tmp_path);entry=accepted_pair(tmp_path)
    registry=tmp_path/'registry.json';save(registry,{'fields':[entry]})
    worker=StreamCoordinator(tmp_path,{**settings(),'physics_registry':str(registry),'minimum_native_positive':100,'minimum_native_zero':100},executor=lambda *args:Process())
    worker.tick();assert len(worker.state['accepted_pairs'])==1 and len(worker.state['batches'])==1
    doc=json.loads(Path(worker.state['batches'][0]['manifest']).read_text())
    simulated=[r for r in doc['images'] if r['provenance']=='direct_physics']
    assert simulated and len(simulated)/len(doc['images'])<=.25
    assert doc['metadata']['recorded_background_hybrid_training']=='blocked until independently validated'
    assert all(r['sha256'] in worker.state['consumed'] for r in simulated)
    assert len(worker.state['consumed'])<64

def test_synthetic_dominant_stream_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='25%'):StreamCoordinator(tmp_path,{'maximum_synthetic_fraction':.8})

def test_nonzero_kl_gradients_cannot_be_reported_as_task_reward_learning(tmp_path):
    from aerorozum.stream_training import reward_signal_diagnostics
    path=tmp_path/'metrics.jsonl'
    path.write_text(json.dumps({'step':1,'grad_norm':.4,'reward':1.,'frac_reward_zero_std':1.})+'\n')
    result=reward_signal_diagnostics(path)
    assert result['measured_steps']==1 and result['task_signal_verified'] is False
    with path.open('a') as f:f.write(json.dumps({'step':2,'frac_reward_zero_std':.75})+'\n')
    assert reward_signal_diagnostics(path)['task_signal_verified'] is True

def test_sft_utility_survives_zero_variance_or_regressing_grpo():
    from aerorozum.stream_training import checkpoint_utility
    def score(f2):return {'micro_f2':f2,'valid_json_fraction':1.,'negative_false_alarm_rate':0.}
    assert checkpoint_utility(score(.4),score(.7),score(.75),{'task_signal_verified':False})['selected_stage']=='sft'
    assert checkpoint_utility(score(.4),score(.7),score(.6),{'task_signal_verified':True})['selected_stage']=='sft'
    result=checkpoint_utility(score(.4),score(.7),score(.8),{'task_signal_verified':True})
    assert result['selected_stage']=='grpo' and result['rl_improvement_claimed'] is False

def test_completed_pass_queues_sft_evaluation_and_promotes_useful_sft(tmp_path):
    real_index(tmp_path);worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:Process());worker.tick()
    job=worker.state['jobs'][0];manifest=json.loads(Path(job['manifest']).read_text())
    def stage(name,steps):
        adapter=tmp_path/'training'/name/'seed-17/adapter';adapter.mkdir(parents=True)
        (adapter/'adapter_config.json').write_text('{}');(adapter/'adapter_model.safetensors').write_bytes(b'test-only')
        return {'status':'completed','stage':'stream_'+name,'manifest_hash':digest(manifest),'seed':17,
            'optimizer_updates':steps,'nonzero_gradient_checks':steps,'adapter':str(adapter.relative_to(tmp_path))}
    def score(f2):return {'status':'completed','examples':8,'micro_f2':f2,'valid_json_fraction':1.,'negative_false_alarm_rate':0.}
    result={'status':'completed','batch_id':job['batch_id'],'seed':17,'sft':stage('sft',32),'grpo':stage('grpo',16),
        'baseline':score(.4),'evaluation':score(.6),'promoted':True}
    result['adapter']=result['grpo']['adapter'];save(Path(job['output'])/'result.json',result)
    worker.manage_jobs();evaluation=worker.state['evaluation_jobs'][0]
    assert job['status']=='completed' and evaluation['status']=='running' and not worker.state['promoted']
    save(Path(evaluation['output'])/'summary.json',score(.7));worker.manage_jobs()
    assert evaluation['status']=='completed' and worker.state['promoted']['17']['stage']=='sft'
    assert worker.state['promoted']['17']['rl_improvement_claimed'] is False
