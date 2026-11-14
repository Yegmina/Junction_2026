import json
from pathlib import Path
import numpy as np
import pytest
from aerorozum.environment import certify_observability
from aerorozum.generation import plan_generation,generate_dataset,generate_demo_dataset
from aerorozum.fast_fields import file_hash


def test_missing_accepted_fields_blocks_exact_research_counts(tmp_path):
    frozen={'config_hash':'frozen','counts':{'images':30000},'campaigns':[{'id':'c1','split':'train','seed':1}]}
    result=generate_dataset(frozen,{'config_hash':'frozen','fields':[]},tmp_path)
    assert result['status']=='blocked' and not result['research_accepted']
    assert result['requested_images']==30000
    assert not list(tmp_path.glob('**/*.png'))
    assert not (tmp_path/'manifest.json').exists()
    with pytest.raises(ValueError,match='reduce'):
        generate_dataset(frozen,{},tmp_path,count=8)


def test_missing_registry_is_a_blocked_stage(tmp_path):
    result=plan_generation({'config_hash':'x','counts':{'images':30000},'campaigns':[]},tmp_path/'missing.json')
    assert result['status']=='blocked' and result['requested_images']==30000


def test_demo_is_explicitly_unaccepted_and_exact_count(tmp_path):
    result=generate_demo_dataset(tmp_path,count=8,seed=12)
    assert result['status']=='demo' and result['images']==8 and not result['research_accepted']
    manifest=json.loads(Path(result['manifest']).read_text())
    assert len(manifest['images'])==8
    for row in manifest['images']:
        assert row['split']=='demo' and row['provenance']=='analytic_demo_not_research'
        assert not row['research_accepted'] and Path(row['image']).is_file()


def fixture_registry(tmp_path):
    cfg={'campaign':{'width_m':64,'height_m':64},'camera':{'width':128,'height':128,'hfov_deg':20,'vfov_deg':20}}
    axis=np.linspace(0,64,64);xx,yy=np.meshgrid(axis,axis)
    baseline=np.ones((2,64,64))*285
    present=baseline+np.exp(-((xx-32)**2+(yy-32)**2)/(2*.8**2))[None,:,:]
    for name,temp,targets in [('present',present,[{'id':'target','x_m':32,'y_m':32,'radius_m':.8}]),('absent',baseline,[])]:
        path=tmp_path/f'{name}.npz'
        np.savez_compressed(path,times_s=[0.,86400.],x_m=axis,y_m=axis,temperature_k=temp,moisture=np.ones_like(temp)*.2)
        path.with_suffix('.json').write_text(json.dumps({'fidelity':'direct_physics_unit_fixture_only','backend':'dumux_unit_fixture_only','field_sha256':file_hash(path),'research_accepted':True,'validation_state':'numerically_accepted','paired_environment_hash':'paired','domain':{'width_m':64,'height_m':64}}))
        path.with_suffix('.truth.json').write_text(json.dumps({'targets':targets}))
    cert=tmp_path/'certificate.json'
    certify_observability({**cfg,'thermal_field_path':str(tmp_path/'present.npz')},cert,seed=3)
    acceptance=tmp_path/'acceptance.json';acceptance.write_text(json.dumps({'accepted':True,'status':'numerically_accepted_pair','research_accepted':True,'kind':'unit_test_fixture_not_production',
        'present_sha256':file_hash(tmp_path/'present.npz'),'absent_sha256':file_hash(tmp_path/'absent.npz')}))
    frozen={'config_hash':'frozen','counts':{'images':8},'campaigns':[{'id':'c1','split':'train','seed':3}]}
    registry={'config_hash':'frozen','fields':[{'campaign_id':'c1','split':'train','status':'accepted','field_path':str(tmp_path/'present.npz'),
        'counterfactual_path':str(tmp_path/'absent.npz'),'observability_manifest_path':str(cert),'acceptance_report_path':str(acceptance),
        'background_group':'bg-train','paired_environment_hash':'paired'}]}
    return cfg,frozen,registry


def test_accepted_generation_exact_count_and_resume(tmp_path):
    cfg,frozen,registry=fixture_registry(tmp_path)
    assert plan_generation(frozen,registry)['status']=='ready'
    result=generate_dataset(frozen,registry,tmp_path/'dataset',cfg=cfg,artifact_root=tmp_path)
    assert result['status']=='completed' and result['images']==8
    manifest=json.loads(Path(result['manifest']).read_text())
    assert all(row['provenance']=='direct_physics' and row['split']=='train' and not Path(row['image']).is_absolute() for row in manifest['images'])
    from aerorozum.training import load_training_manifest
    selected,protocol=load_training_manifest(result['manifest'],'sft',root=tmp_path)
    assert len(selected)==8 and protocol['synthetic_images']==8
    hashes=[row['image_sha256'] for row in manifest['images']]
    again=generate_dataset(frozen,registry,tmp_path/'dataset',cfg=cfg,artifact_root=tmp_path)
    assert again['images']==8
    assert len((tmp_path/'dataset/images.jsonl').read_text().splitlines())==8
    assert [r['image_sha256'] for r in json.loads(Path(again['manifest']).read_text())['images']]==hashes


def test_unaccepted_counterfactual_cannot_generate_research_labels(tmp_path):
    cfg,frozen,registry=fixture_registry(tmp_path)
    (tmp_path/'absent.json').write_text(json.dumps({'research_accepted':False}))
    result=generate_dataset(frozen,registry,tmp_path/'blocked',cfg=cfg)
    assert result['status']=='blocked' and not result['research_accepted']


def test_background_split_leakage_is_rejected(tmp_path):
    cfg,frozen,registry=fixture_registry(tmp_path)
    frozen['campaigns'].append({'id':'c2','split':'validation','seed':4})
    registry['fields'].append({**registry['fields'][0],'campaign_id':'c2','split':'validation'})
    result=plan_generation(frozen,registry)
    assert result['status']=='blocked'
    assert any('crosses' in error for error in result['problems'])

def test_generation_accepts_independent_adaptive_time_grids_with_common_coverage(tmp_path):
    cfg,frozen,registry=fixture_registry(tmp_path)
    path=tmp_path/'absent.npz'
    with np.load(path) as source:
        values={key:source[key] for key in source.files}
    values.update(times_s=np.array([0.,21600.,86400.]),temperature_k=np.repeat(values['temperature_k'][:1],3,axis=0),moisture=np.repeat(values['moisture'][:1],3,axis=0))
    np.savez_compressed(path,**values)
    metadata=json.loads(path.with_suffix('.json').read_text());metadata['field_sha256']=file_hash(path);path.with_suffix('.json').write_text(json.dumps(metadata))
    certificate=tmp_path/'acceptance.json';report=json.loads(certificate.read_text());report['absent_sha256']=file_hash(path);certificate.write_text(json.dumps(report))
    assert plan_generation(frozen,registry)['status']=='ready'

def test_generation_requires_hash_bound_pair_acceptance(tmp_path):
    cfg,frozen,registry=fixture_registry(tmp_path)
    report=tmp_path/'acceptance.json';report.write_text(json.dumps({'research_accepted':True}))
    result=plan_generation(frozen,registry)
    assert result['status']=='blocked' and any('certificate' in message for message in result['problems'])
