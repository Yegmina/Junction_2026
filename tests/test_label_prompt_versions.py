import asyncio
import hashlib
import json
from PIL import Image
from aerorozum import labelling
from aerorozum.label_prompts import PROMPT_V1, PROMPT_V2


class Provider:
    def __init__(self):
        self.cfg={};self.calls=[]

    async def generate(self,role,prompt,images,model):
        self.calls.append(prompt)
        return {'detections':[]}


def frame(tmp_path,index):
    path=tmp_path/f'frame-{index}.png'
    Image.new('RGB',(32,32),(index*30,50,60)).save(path)
    return {'id':str(index),'image':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'split':'train','source':{'session_id':'s','generation':'123'},'frame_index':index}


def test_prompt_change_reuses_paid_old_labels_without_renaming_them(tmp_path,monkeypatch):
    row=frame(tmp_path,1);provider=Provider();root=tmp_path/'labels'
    monkeypatch.setattr(labelling,'PROMPT',PROMPT_V1)
    asyncio.run(labelling.label_frames([row],root,provider,concurrency=1))
    old_key=labelling.inference_cache_key(row,row['sha256'],provider)
    monkeypatch.setattr(labelling,'PROMPT',PROMPT_V2)
    new_key=labelling.inference_cache_key(row,row['sha256'],provider)
    assert old_key!=new_key
    second=Provider()
    result=asyncio.run(labelling.label_frames([row],root,second,concurrency=1))
    assert result['labelled']==1 and second.calls==[]
    assert not (root/'results'/(new_key+'.json')).exists()
    saved=json.loads((root/'results'/(old_key+'.json')).read_text())
    assert saved['label_prompt_version']=='thermal-v1-conservative'
    assert saved['label_prompt_sha256']==hashlib.sha256(PROMPT_V1.encode()).hexdigest()
    audit=json.loads(next((root/'prompt-audit').glob('*.json')).read_text())
    assert audit['cache_key']==old_key and audit['label_prompt_version']=='thermal-v1-conservative'


def test_new_frames_store_new_prompt_and_keep_low_confidence_uncertain(tmp_path,monkeypatch):
    row=frame(tmp_path,2);provider=Provider();root=tmp_path/'labels'
    monkeypatch.setattr(labelling,'PROMPT',PROMPT_V2)
    async def generate(*args,**kwargs):
        provider.calls.append(args[1])
        return {'detections':[{'confidence':.35,'bbox':{'x_min':10,'y_min':10,'x_max':30,'y_max':30}}]}
    provider.generate=generate
    result=asyncio.run(labelling.label_frames([row],root,provider,concurrency=1))
    saved=json.loads(next((root/'results').glob('*.json')).read_text())
    assert provider.calls==[PROMPT_V2]
    assert saved['label_prompt_version']=='thermal-v2-recall-calibrated'
    assert saved['label_prompt_sha256']==hashlib.sha256(PROMPT_V2.encode()).hexdigest()
    assert saved['boxes']==[] and len(saved['uncertain_boxes'])==1
    assert saved['zero_candidates'] is False and result['training_frames']==0


def test_airborne_worker_does_not_pay_for_unknown_or_ground_frames(tmp_path):
    provider=Provider();provider.cfg['require_airborne_verified']=True
    unknown=frame(tmp_path,3);ground=frame(tmp_path,4)
    ground.update(airborne_eligible=False,eligibility={'status':'ground','airborne_eligible':False})
    result=asyncio.run(labelling.label_frames([unknown,ground],tmp_path/'labels',provider,concurrency=1))
    assert result['failed']==2 and provider.calls==[]
