import importlib.util
import json
from pathlib import Path
import pytest
from aerorozum.training import detection_score
spec=importlib.util.spec_from_file_location('validate_real_transfer',Path(__file__).parents[1]/'scripts/validate_real_transfer.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
B={'x_min':100,'y_min':200,'x_max':150,'y_max':260}
def row():return {'id':'x__zone7__1','sha256':'a'*64,'image':'a.jpg','boxes':[{'bbox':B}],'provenance':'original_manual'}
def test_raw_saved_before_scoring_and_resume(tmp_path,monkeypatch):
 r=row();calls=[]
 def infer(r):calls.append(1);return {'text':json.dumps({'detections':[{'bbox':B}]})}
 old=v.score_boxes
 def checked(*args,**kwargs):
  assert len(list((tmp_path/'raw/base').glob('*.json')))==1
  return old(*args,**kwargs)
 monkeypatch.setattr(v,'score_boxes',checked)
 assert v.evaluate_row(r,'base','p',tmp_path,infer)['f2']==1
 assert v.evaluate_row(r,'base','p',tmp_path,infer)['f2']==1
 assert len(calls)==1

def test_invalid_output_preserved_and_counts_miss(tmp_path):
 s=v.evaluate_row(row(),'base','p',tmp_path,lambda r:{'text':'not valid json'})
 assert s['fn']==1 and s['f2']==0 and not s['valid_json']
 assert json.loads(next((tmp_path/'raw/base').glob('*.json')).read_text())['text']=='not valid json'
 assert v.aggregate([s])['negative_image_false_alarm_rate'] is None

@pytest.mark.parametrize('pred',[[B],[B,B],[],[dict(x_min=0,y_min=0,x_max=1000,y_max=1000)]])
def test_matches_training_scorer(pred):
 a=v.score_boxes(pred,[B]);b=detection_score(pred,[B])
 assert {k:a[k] for k in b}==b

def test_freeze_only_zone7_and_verify_hash(tmp_path):
 image=tmp_path/'a.jpg';image.write_bytes(b'actualbytes');r={**row(),'split':'validation','zone':'zone7','sha256':v.file_sha(image)}
 source=tmp_path/'source.json';source.write_text(json.dumps({'images':[r,{**r,'zone':'zone8','split':'test'}]}));train=tmp_path/'train.json';train.write_text(json.dumps({'images':[]}))
 output=tmp_path/'frozen.json';d=v.freeze_validation(source,train,tmp_path,output,expected_count=1)
 assert len(d['images'])==1 and d['images'][0]['zone']=='zone7'
 assert d['metadata']['positive_only'] is True
 train.write_text(json.dumps({'images':[r]}))
 with pytest.raises(ValueError,match='leaked'):v.freeze_validation(source,train,tmp_path,output,expected_count=1)

def test_freeze_rejects_zone_mismatch_and_modified_image(tmp_path):
 (tmp_path/'a.jpg').write_bytes(b'notthesame');r={**row(),'split':'validation','zone':'zone7'}
 source=tmp_path/'source.json';source.write_text(json.dumps({'images':[r]}));train=tmp_path/'train.json';train.write_text('{"images":[]}')
 with pytest.raises(ValueError,match='checksum'):v.freeze_validation(source,train,tmp_path,tmp_path/'f.json',expected_count=1)
