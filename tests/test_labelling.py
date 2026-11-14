import asyncio
import hashlib
import copy
import json
from pathlib import Path
import subprocess
import pytest
from PIL import Image
from aerorozum.labelling import MODEL,PROMPT,classify_response,sample_indices,freeze_session_splits,extract_video_samples,label_frames,label_video_sources
from aerorozum.common import digest

def detection(confidence):
    return {'label':'provisional mine-like thermal candidate','confidence':confidence,'bbox':{'x_min':100,'y_min':100,'x_max':300,'y_max':300},'evidence':'compact signature'}

class Provider:
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=[]
    async def generate(self,role,prompt,images,model):
        assert model==MODEL;self.calls.append(str(images[0]));return next(self.outputs)

def rows(tmp_path,count):
    result=[]
    for i in range(count):
        image=tmp_path/f'{i}.jpg';Image.new('RGB',(64,64),(i*20,50,60)).save(image)
        result.append({'id':str(i),'image':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'width':64,'height':64,'split':'train','source':{'session_id':'session'}})
    return result

def test_threshold_and_weak_candidates_are_not_backgrounds():
    result=classify_response({'detections':[detection(.6),detection(.59),detection(0)]})
    assert len(result['accepted'])==1 and len(result['uncertain'])==2 and not result['zero_candidates']
    assert classify_response({'detections':[]})['zero_candidates']
    with pytest.raises(ValueError):classify_response({})
    with pytest.raises(ValueError):classify_response({'detections':[detection(float('nan'))]})

def test_source_sampling_never_fabricates_duplicate_frames():
    assert sample_indices([0,.1,.2,.3,.4,.5,.6,.7,.8,.9],5)==[0,2,4,6,8]
    assert sample_indices([0,.5,1],5)==[0,1,2]
    with pytest.raises(ValueError):sample_indices([0,0,.2])

def test_session_split_is_immutable_before_frame_selection(tmp_path):
    path=tmp_path/'splits.json'
    assert freeze_session_splits(['recording'],path,assignments={'recording':'test'})['recording']=='test'
    with pytest.raises(ValueError,match='leak'):freeze_session_splits(['recording'],path,assignments={'recording':'train'})

def test_label_provenance_exports_and_resume_do_not_repeat_paid_requests(tmp_path):
    source=tmp_path/'images';source.mkdir();images=rows(source,3)
    provider=Provider([{'detections':[detection(.8)]},{'detections':[detection(.2)]},{'detections':[]}])
    result=asyncio.run(label_frames(images,tmp_path/'labels',provider,concurrency=1))
    assert result['labelled']==3 and result['positive_frames']==1 and result['uncertain_frames']==1 and result['zero_candidate_frames']==1
    manifest=json.loads((tmp_path/'labels/manifest.json').read_text())
    assert len(manifest['images'])==2
    assert all(r['provenance']=='user_accepted_model' and not r['human_reviewed'] for r in manifest['images'])
    assert len(list((tmp_path/'labels/export/labelme').glob('*.json')))==3
    coco=json.loads((tmp_path/'labels/export/coco.json').read_text());assert len(coco['annotations'])==1
    recent=json.loads((tmp_path/'labels/recent.json').read_text())
    assert len(recent['images'])==3
    assert all(Path(row['annotated']).is_file() for row in recent['images'])
    assert all('raw_response' not in row and 'model_output' not in row and 'prompt' not in row for row in recent['images'])
    second=Provider([]);again=asyncio.run(label_frames(images,tmp_path/'labels',second,concurrency=2))
    assert again['labelled']==3 and not second.calls

def test_wrong_model_refuses_before_any_request(tmp_path):
    provider=Provider([])
    with pytest.raises(ValueError,match='fallback'):asyncio.run(label_frames([],tmp_path,provider,model='another-model'))
    assert not provider.calls

def test_mixed_uncertainty_is_preserved_and_not_implicitly_negative(tmp_path):
    source=tmp_path/'images';source.mkdir()
    output={'detections':[detection(.9),{**detection(.1),'bbox':{'x_min':600,'y_min':600,'x_max':800,'y_max':800}}]}
    result=asyncio.run(label_frames(rows(source,1),tmp_path/'labels',Provider([output]),concurrency=1))
    assert result['positive_frames']==result['uncertain_frames']==1
    assert result['zero_candidate_frames']==result['training_frames']==0

def test_streaming_frames_start_labelling_before_producer_finishes(tmp_path):
    source=tmp_path/'images';source.mkdir();images=rows(source,2);provider=Provider([{'detections':[]},{'detections':[]}])
    async def stream():
        yield images[0]
        for _ in range(20):
            if provider.calls:break
            await asyncio.sleep(.001)
        assert provider.calls
        yield images[1]
    assert asyncio.run(label_frames(stream(),tmp_path/'labels',provider,concurrency=1))['labelled']==2

def make_video(path):
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc=size=64x64:rate=10:duration=1','-c:v','mjpeg','-threads','1','-y',str(path)],check=True)

def test_native_extraction_matches_receipt_clock_and_rejects_misalignment(tmp_path):
    video=tmp_path/'thermal-0000.avi';make_video(video)
    timestamps=tmp_path/'thermal-timestamps.jsonl'
    timestamps.write_text(''.join(json.dumps({'received_at_ns':int((100+i*.1)*1e9),'segment':0})+'\n' for i in range(10)))
    result=extract_video_samples(video,tmp_path/'frames',session_id='session',timestamps_path=timestamps)
    assert [r['frame_index'] for r in result]==[0,2,4,6,8]
    assert all(r['timestamp_s'] is not None and r['time_basis'].startswith('jetson') for r in result)
    timestamps.write_text(timestamps.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='one-to-one'):extract_video_samples(video,tmp_path/'bad',session_id='session',timestamps_path=timestamps)

def test_bad_source_quarantines_and_next_source_continues(tmp_path):
    video=tmp_path/'good.avi';make_video(video)
    sources=[{'path':str(tmp_path/'missing.avi'),'session_id':'one'},{'path':str(video),'session_id':'two'}]
    provider=Provider([{'detections':[]}]*5)
    result=asyncio.run(label_video_sources(sources,tmp_path/'run',provider=provider,concurrency=1))
    assert result['status']=='partial' and result['failed_sources']==1 and result['labelled']==5
    manifest=json.loads((tmp_path/'run/sampling-manifest.json').read_text())
    assert len(manifest['failed_sources'])==1

class Store:
    def __init__(self):self.data={};self.fail_write=False;self.fail_read=False
    def get(self,key,kind='results'):
        if self.fail_read:raise RuntimeError('cloud read unavailable')
        return copy.deepcopy(self.data.get((kind,key)))
    def put(self,key,record,kind='results'):
        if self.fail_write:raise RuntimeError('cloud upload unavailable')
        self.data.setdefault((kind,key),copy.deepcopy(record));return 'gs://test/'+kind+'/'+key
    def write_frame(self,key,record,raw_log=None):
        result={'result':self.put(key,record),'labels':self.put(key,record,'labels')}
        if raw_log is not None:result['raw_log']=self.put(key,raw_log,'logs')
        return result

class LoggedProvider(Provider):
    cfg={'thinking_budget':0,'api_version':'v1beta','reserve_output_tokens':2048}
    def response_record(self,*args,**kwargs):
        return {'raw_text':'{"detections":[]}','model':MODEL,'prompt':'actual logged prompt','usage':{'total_token_count':10},'elapsed_s':.1}

def test_cloud_hit_prevents_paid_generation_and_retains_raw_logs(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1);store=Store()
    first=LoggedProvider([{'detections':[]}])
    result=asyncio.run(label_frames(images,tmp_path/'one',first,record_store=store,concurrency=1))
    assert result['cloud_sync']['uploaded']==1
    assert any(kind=='logs' for kind,key in store.data)
    second=LoggedProvider([])
    again=asyncio.run(label_frames(images,tmp_path/'two',second,record_store=store,concurrency=1))
    assert not second.calls and again['cloud_sync']['cloud_cache_hits']==1

def test_failed_upload_retries_from_local_paid_result_without_api(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1);store=Store();store.fail_write=True
    first=LoggedProvider([{'detections':[]}])
    result=asyncio.run(label_frames(images,tmp_path/'labels',first,record_store=store,concurrency=1))
    assert result['status']=='upload_pending' and result['cloud_sync']['upload_pending']==1
    store.fail_write=False;store.fail_read=True;second=LoggedProvider([])
    again=asyncio.run(label_frames(images,tmp_path/'labels',second,record_store=store,concurrency=1))
    assert again['status']=='completed' and not second.calls

def test_cloud_read_failure_never_triggers_a_paid_request(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1);store=Store();store.fail_read=True
    provider=LoggedProvider([])
    result=asyncio.run(label_frames(images,tmp_path/'labels',provider,record_store=store,concurrency=1))
    assert result['failed']==1 and not provider.calls

def test_paid_response_survives_interruption_before_label_record(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1)
    first=LoggedProvider([{'detections':[]}])
    asyncio.run(label_frames(images,tmp_path/'labels',first,concurrency=1))
    for path in (tmp_path/'labels/results').glob('*.json'):path.unlink()
    second=LoggedProvider([])
    result=asyncio.run(label_frames(images,tmp_path/'labels',second,concurrency=1))
    assert result['labelled']==1 and not second.calls

def test_identical_pixels_share_one_paid_request_within_parallel_batch(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1)
    images.append({**images[0],'id':'second-frame','frame_index':2})
    provider=LoggedProvider([{'detections':[]}]);store=Store()
    result=asyncio.run(label_frames(images,tmp_path/'labels',provider,record_store=store,concurrency=2))
    assert result['labelled']==2 and len(provider.calls)==1

def test_invalid_paid_schema_is_cached_and_raw_log_uploaded_without_rebuy(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1);store=Store()
    provider=LoggedProvider([{'detections':[{'confidence':.8,'bbox':{}}]}])
    result=asyncio.run(label_frames(images,tmp_path/'labels',provider,record_store=store,concurrency=1))
    assert result['failed']==1 and any(kind=='logs' for kind,key in store.data)
    second=LoggedProvider([])
    again=asyncio.run(label_frames(images,tmp_path/'labels',second,record_store=store,concurrency=1))
    assert again['failed']==1 and not second.calls

def test_owned_provider_forces_beta_zero_thinking_and_actual_concurrency(tmp_path,monkeypatch):
    from aerorozum import providers
    seen=[]
    class Owned(LoggedProvider):
        def __init__(self,cfg,output_dir):seen.append(cfg);super().__init__([{'detections':[]}]);self.cfg=cfg
        async def close(self):pass
    monkeypatch.setattr(providers,'GeminiProvider',Owned)
    image_dir=tmp_path/'images';image_dir.mkdir()
    result=asyncio.run(label_frames(rows(image_dir,1),tmp_path/'labels',concurrency=3))
    assert result['labelled']==1
    assert seen[0]['thinking_budget']==0 and seen[0]['api_version']=='v1beta' and seen[0]['concurrency']==3

def test_old_key_cloud_archive_prevents_rebuying_legacy_crops(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,1)
    asyncio.run(label_frames(images,tmp_path/'old',Provider([{'detections':[]}]),concurrency=1))
    original=json.loads(next((tmp_path/'old/results').glob('*.json')).read_text())
    original.pop('cache_key',None);original.pop('raw_response',None)
    key=digest([images[0]['id'],images[0]['sha256'],MODEL,.6,PROMPT]);store=Store()
    store.data['results',key]=original;store.data['logs',key]={'legacy':True,'raw_text_available':False,'usage':{'total_token_count':10}}
    provider=LoggedProvider([])
    result=asyncio.run(label_frames(images,tmp_path/'new',provider,record_store=store,concurrency=1))
    assert result['labelled']==1 and not provider.calls and result['cloud_sync']['cloud_cache_hits']==1
    migrated=json.loads(next((tmp_path/'new/results').glob('*.json')).read_text())
    assert migrated['cache_migration']['source_key']==key

def test_recent_index_stays_bounded_without_private_logs(tmp_path):
    image_dir=tmp_path/'images';image_dir.mkdir();images=rows(image_dir,14)
    asyncio.run(label_frames(images,tmp_path/'labels',LoggedProvider([{'detections':[]}]*14),concurrency=1))
    recent=json.loads((tmp_path/'labels/recent.json').read_text())['images']
    assert len(recent)==12 and recent[0]['id']=='2' and recent[-1]['id']=='13'
    assert all('raw_response' not in row for row in recent)

def test_frame_failure_persisted_before_other_inflight_request_finishes(tmp_path):
 source=tmp_path/'images';source.mkdir();images=rows(source,2);output=tmp_path/'labels'
 class BlockedProvider:
  async def generate(self,role,prompt,paths,model):
   if paths[0].name=='0.jpg':raise RuntimeError('GCS cache read failed: HTTP 401; paid inference was not issued')
   for _ in range(100):
    if (output/'failure-journal.jsonl').exists():break
    await asyncio.sleep(.001)
   failure=json.loads((output/'failure-journal.jsonl').read_text().splitlines()[0])
   assert failure['id']=='0' and failure['category']=='cloud_cache_authentication'
   assert failure['row']['sha256']==images[0]['sha256']
   assert len(list((output/'failures').glob('*.json')))==1
   return {'detections':[]}
 result=asyncio.run(label_frames(images,output,BlockedProvider(),concurrency=2))
 assert result['failed']==1 and result['labelled']==1
 states=[json.loads(p.read_text())['status'] for p in (output/'checkpoints').glob('*.json')]
 assert sorted(states)==['completed','failed']
 progress=json.loads((output/'progress.json').read_text())
 assert progress['failure_reasons']=={'cloud_cache_authentication':1}

def test_already_uploaded_exact_frame_binding_resumes_without_cloud_rewrite(tmp_path):
 source=tmp_path/'images';source.mkdir();images=rows(source,1);store=Store();first=LoggedProvider([{'detections':[]}])
 asyncio.run(label_frames(images,tmp_path/'labels',first,record_store=store,concurrency=1));store.fail_write=True;second=LoggedProvider([])
 resumed=asyncio.run(label_frames(images,tmp_path/'labels',second,record_store=store,concurrency=1))
 assert resumed['labelled']==1 and not second.calls and resumed['cloud_sync']['resumed_verified']==1
 assert resumed['cloud_sync']['upload_pending']==0

def test_verified_sampling_cache_resume_preserves_saved_frame_bytes(tmp_path,monkeypatch):
 import aerorozum.labelling as module
 video=tmp_path/'source.avi';make_video(video);folder=tmp_path/'frames'
 first=extract_video_samples(video,folder,session_id='one');actual_probe=module._probe(video);before={r['id']:Path(r['image']).stat().st_mtime_ns for r in first}
 def no_decode(*args,**kwargs):raise AssertionError('verified frames must not be decoded/overwritten on resume')
 monkeypatch.setattr(module,'_probe',lambda *a:actual_probe);monkeypatch.setattr(module.subprocess,'Popen',no_decode)
 again=extract_video_samples(video,folder,session_id='one')
 assert again==first and all(Path(r['image']).stat().st_mtime_ns==before[r['id']] for r in again)
