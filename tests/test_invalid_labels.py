import asyncio,copy,hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from aerorozum.common import save,digest
from aerorozum.invalid_labels import normalize_box_aliases,recover_invalid
from aerorozum.labelling import label_frames,inference_cache_key,MODEL,PROMPT


def test_alias_repairs_only_explicit_unambiguous_names():
    original={'detections':[{'bbox':{'x_min':10,'y_min':20,'xmax':30,'y_max':40},'confidence':.2}]}
    fixed,changes=normalize_box_aliases(original)
    assert fixed['detections'][0]['bbox']['x_max']==30 and len(changes)==1
    assert 'xmax' in original['detections'][0]['bbox']
    for malformed in ({},{'detections':[{'bbox':{'x_min':10,'ymax':20,'x_max':30,'y_max':40},'confidence':.2}]},{'detections':[{'bbox':{'x_min':10,'y_min':20,'total_width':20,'y_max':40},'confidence':.8}]}):
        with pytest.raises(ValueError):normalize_box_aliases(malformed)


class Provider:
    cfg={'api_version':'v1beta','thinking_budget':0}
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=[];self.cfg=dict(self.cfg);self.records={}
    async def generate(self,role,prompt,images,model):
        self.calls.append(role);output=next(self.outputs);self.records[role]={'output':output,'model':MODEL,'image_hashes':[hashlib.sha256(Path(images[0]).read_bytes()).hexdigest()]};return copy.deepcopy(output)
    def response_record(self,role,prompt,images,model):return self.records.get(role)


def setup(tmp_path,output):
    frame=tmp_path/'labelling/source/frames/one/f.jpg';frame.parent.mkdir(parents=True);Image.new('RGB',(32,32),(70,80,90)).save(frame)
    row={'id':'native1','image':str(frame),'sha256':hashlib.sha256(frame.read_bytes()).hexdigest(),'split':'train','source':{'session_id':'s','generation':'g'}}
    save(frame.parent/'sampling.json',{'images':[row]});labels=tmp_path/'labelling/robotics-er2-5fps/labels';provider=Provider([])
    # A historical paid failure predates the inline alias normalizer. Construct
    # that archived state directly; new calls now repair safe aliases immediately.
    key=inference_cache_key(row,row['sha256'],provider)
    archived={'cache_key':key,'model':MODEL,'sha256':row['sha256'],'source':row['source'],
              'status':'invalid_model_response','model_output':copy.deepcopy(output),
              'raw_response':{'model':MODEL,'prompt':PROMPT,'raw_text':json.dumps(output),'output':copy.deepcopy(output)}}
    save(labels/'results'/(key+'.json'),archived)
    save(labels/'paid-responses'/(key+'.json'),archived)
    return labels,row


def test_alias_recovery_preserves_original_and_reuses_replacement(tmp_path):
    bad={'detections':[{'bbox':{'x_min':10,'y_min':20,'xmax':30,'y_max':40},'confidence':.8}]};labels,row=setup(tmp_path,bad);original=next((labels/'results').glob('*.json'));before=original.read_bytes();provider=Provider([])
    result=asyncio.run(recover_invalid(provider,tmp_path));assert result['results'][0]['status']=='recovered' and not provider.calls
    assert original.read_bytes()==before
    assert asyncio.run(label_frames([row],labels,provider,concurrency=1))['positive_frames']==1
    assert original.read_bytes()==before and not provider.calls


def test_invalid_retry_is_bounded_and_never_becomes_background(tmp_path):
    labels,row=setup(tmp_path,{});provider=Provider([{}]);first=asyncio.run(recover_invalid(provider,tmp_path));assert len(provider.calls)==1
    second=asyncio.run(recover_invalid(provider,tmp_path));assert len(provider.calls)==1
    assert len(first['results'])==len(second['results'])==1
    assert all(json.loads(p.read_text()).get('zero_candidates') is not True for p in (labels/'results').glob('*.json'))


def test_revision_leaves_existing_cache_identity_unchanged():
    provider=Provider([]);row={'source':{'generation':'1'}};sha='a'*64
    old=digest({'model':MODEL,'image_sha256':sha,'prompt':PROMPT,'generation_config':{'temperature':.1,'max_output_tokens':2048,'api_version':'v1beta','thinking_budget':0},'source_generation':'1'})
    assert inference_cache_key(row,sha,provider)==old
    provider.cfg['inference_revision']='invalid-retry1:key'
    assert inference_cache_key(row,sha,provider)!=old
