import asyncio,copy,json
from pathlib import Path
import pytest
from PIL import Image
from aerorozum.common import save
from aerorozum.fast_fields import file_hash
from aerorozum.prompt_comparison import freeze_protocol,aggregate,candidates,run,VERSIONS

B={'x_min':111,'y_min':222,'x_max':333,'y_max':444}
C={'x_min':600,'y_min':600,'x_max':700,'y_max':700}

def inputs(root,count=88):
 rows=[]
 for i in range(count):
  p=root/'images'/f'{i}.png';p.parent.mkdir(exist_ok=True);Image.new('RGB',(32,32),(i%255,i*3%255,i*7%255)).save(p)
  rows.append({'id':str(i),'image':str(p.relative_to(root)),'sha256':file_hash(p),'zone':'zone7','split':'validation','provenance':'original_manual','boxes':[{'bbox':B}]})
 rows.append({'id':'sealed','zone':'zone9','split':'test','image':'never-open-this-file','boxes':[{'bbox':C}]})
 save(root/'datasets/real/manifest.json',{'images':rows});return rows[:-1]


def test_frozen_protocol_preserves_prompts_and_sealed_zones(tmp_path,monkeypatch):
 inputs(tmp_path);out=tmp_path/'evaluation';p=freeze_protocol(tmp_path,out)
 assert len(p['images'])==88 and not any(r['zone']!='zone7' for r in p['images'])
 monkeypatch.setitem(__import__('aerorozum.prompt_comparison',fromlist=['PROMPTS']).PROMPTS,VERSIONS[1],'changed after freeze')
 assert freeze_protocol(tmp_path,out)==p
 p['expected_images']=87;save(out/'protocol.json',p)
 with pytest.raises(ValueError,match='changed'):freeze_protocol(tmp_path,out)


def test_box_metrics_low_confidence_and_no_background_fpr(tmp_path):
 rows=inputs(tmp_path,1);p=freeze_protocol(tmp_path,tmp_path/'eval',expected_images=1)
 records={(rows[0]['id'],VERSIONS[0]):{'status':'completed','model_output':{'detections':[{'bbox':B,'confidence':.4}]}},
          (rows[0]['id'],VERSIONS[1]):{'status':'completed','model_output':{'detections':[{'bbox':B,'confidence':.8},{'bbox':C,'confidence':.9}]}}}
 report=aggregate(p,records);a,b=(report['versions'][v] for v in VERSIONS)
 assert a['at_0_6']['recall']==0 and a['threshold_curve'][3]['recall']==1
 assert b['at_0_6']['tp']==1 and b['at_0_6']['unmatched_boxes_per_image']==1
 assert report['background_false_positive_rate'] is None and report['measured_winner_at_0_6']==VERSIONS[1]
 records.pop((rows[0]['id'],VERSIONS[0]));assert aggregate(p,records)['measured_winner_at_0_6'] is None


def test_invalid_confidence_cannot_be_promoted():
 good,bad=candidates({'detections':[{'bbox':B,'confidence':True},{'bbox':B,'confidence':float('nan')},{'bbox':B,'confidence':.2}]})
 assert bad==2 and len(good)==1 and good[0]['confidence']==.2

def test_strict_whole_frame_and_lossless_alias_diagnostics(tmp_path):
 from aerorozum.prompt_comparison import schema_diagnostic
 rows=inputs(tmp_path,1);protocol=freeze_protocol(tmp_path,tmp_path/'eval',expected_images=1)
 output={'detections':[{'bbox':B,'confidence':.8},{'bbox':{'xmin':600,'ymin':600,'xmax':700,'ymax':700},'confidence':.2}]}
 records={(rows[0]['id'],version):{'status':'completed','model_output':copy.deepcopy(output)} for version in VERSIONS}
 assert aggregate(protocol,records)['versions'][VERSIONS[1]]['at_0_6']['tp']==1
 assert schema_diagnostic(protocol,records)['versions'][VERSIONS[1]]['at_0_6']['tp']==0
 fixed=schema_diagnostic(protocol,records,normalize_aliases=True)
 assert fixed['versions'][VERSIONS[1]]['at_0_6']['tp']==1 and len(fixed['normalization_trace'])==2
 assert 'xmin' in records[(rows[0]['id'],VERSIONS[1])]['model_output']['detections'][1]['bbox']
 records[(rows[0]['id'],VERSIONS[1])]['model_output']['detections'][1]['bbox']={'x_min':600,'y_min':600,'x_max':700,'max':700}
 assert schema_diagnostic(protocol,records,normalize_aliases=True)['versions'][VERSIONS[1]]['at_0_6']['tp']==0


def test_full_paired_run_resumes_without_paid_duplicates_and_archives(tmp_path):
 inputs(tmp_path)
 class Provider:
  issued=0
  def __init__(self):self.saved={};self.calls=[]
  async def generate(self,role,prompt,images,model,**kwargs):
   assert '111' not in prompt and model=='gemini-robotics-er-2-preview';self.calls.append((str(images[0]),prompt));self.issued+=1
   output={'detections':[{'bbox':B,'confidence':.8}] if 'Prioritize finding' in prompt else []};self.saved[(prompt,str(images[0]))]=output;return output
  def response_record(self,role,prompt,images,**kwargs):return {'prompt':prompt,'raw_text':json.dumps(self.saved[(prompt,str(images[0]))]),'usage':{'prompt_token_count':1254}}
 class Store:
  def __init__(self):self.records={};self.logs={}
  def get(self,key,kind='results'):return copy.deepcopy(self.records.get((kind,key)))
  def put(self,key,record,kind='results'):self.records[(kind,key)]=copy.deepcopy(record);return 'gs://unit-test/'+key
  def write_frame(self,key,record,raw_log=None):self.put(key,record);self.logs[key]=copy.deepcopy(raw_log);return {'result':'gs://unit-test/'+key}
 provider=Provider();store=Store();out=tmp_path/'evaluation'
 report=asyncio.run(run(out,provider,store,tmp_path))
 assert report['status']=='completed' and report['paired_images']==88 and provider.issued==176 and len(store.logs)==176
 assert report['measured_winner_at_0_6']==VERSIONS[1]
 assert (out/'report.md').is_file() and (out/'qualitative-selection.json').is_file()
 again=asyncio.run(run(out,provider,store,tmp_path));assert provider.issued==176 and again['versions']==report['versions']
 assert json.loads((tmp_path/'pipeline/leases/prompt-comparison.json').read_text())['status']=='completed'
