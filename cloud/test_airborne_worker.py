import hashlib,json
import pytest
from cloud import airborne_worker as worker

def fixture_plan(tmp_path,monkeypatch):
    from aerorozum import labelling,providers,label_prompts,label_schema
    monkeypatch.setenv('AEROROZUM_DATA_ROOT',str(tmp_path))
    image=tmp_path/'a.jpg';image.write_bytes(b'frozen-test-image')
    audit=tmp_path/'eligibility.json';audit.write_text('{}')
    doc={'status':'airborne_only','eligibility_manifest_sha256':worker.sha(audit),'images':[{'id':'a','image':'a.jpg','sha256':worker.sha(image),'airborne_eligible':True,'eligibility':{'status':'eligible'},'source':{'bucket':'aerorozum-jetson-testdata-smart-bird','object':'a.avi','generation':'123'}}]}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(doc))
    full=tmp_path/'full.json';full.write_text('{}');doc.update(subset='pending_only',policy={'version':'airborne-only-v2-primary-state'},full_manifest='full.json',full_manifest_sha256=worker.sha(full));manifest.write_text(json.dumps(doc))
    plan={k:True for k in ('authorized','laptop_paid_calls_settled','prompt_selection_completed','api_lease_released')}
    plan.update(model=labelling.MODEL,not_before_utc='2020-01-01T00:00:00+00:00',deadline_utc='2099-01-01T00:00:00+00:00',labelling_source_sha256=worker.sha(labelling.__file__),providers_source_sha256=worker.sha(providers.__file__),prompt_sha256=hashlib.sha256(labelling.PROMPT.encode()).hexdigest(),manifest_path=str(manifest),manifest_sha256=worker.sha(manifest),eligibility_manifest_path=str(audit))
    plan.update(label_prompt_version=label_prompts.prompt_metadata(labelling.PROMPT)['label_prompt_version'],label_prompts_source_sha256=worker.sha(label_prompts.__file__))
    plan['label_schema_source_sha256']=worker.sha(label_schema.__file__)
    path=tmp_path/'launch.json';path.write_text(json.dumps(plan));return path,plan,manifest,doc

def test_validates_without_constructing_paid_provider(tmp_path,monkeypatch):
    path,_,_,_=fixture_plan(tmp_path,monkeypatch);assert len(worker.validate(path)[1])==1
    (tmp_path/'a.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='missing or changed'):worker.validate(path)

@pytest.mark.parametrize('gate',['authorized','laptop_paid_calls_settled','prompt_selection_completed','api_lease_released'])
def test_every_handoff_gate_required(tmp_path,monkeypatch,gate):
    path,plan,_,_=fixture_plan(tmp_path,monkeypatch);plan[gate]=False;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='authorization'):worker.validate(path)

def test_ground_frame_rejected_even_when_manifest_hash_matches(tmp_path,monkeypatch):
    path,plan,manifest,doc=fixture_plan(tmp_path,monkeypatch);doc['images'][0]['airborne_eligible']=False;manifest.write_text(json.dumps(doc));plan['manifest_sha256']=worker.sha(manifest);path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='airborne eligibility'):worker.validate(path)

def test_drain_settles_provider_before_cancelling_label_loop(tmp_path,monkeypatch):
    import asyncio
    from aerorozum import providers,labelling,label_archive,cloud_cache
    path,_,_,_=fixture_plan(tmp_path,monkeypatch);events=[]
    credential=tmp_path/'credentials';credential.mkdir();key=credential/'gemini-api-key';key.write_text('test-only');key.chmod(0o600)
    monkeypatch.setenv('CREDENTIALS_DIRECTORY',str(credential))
    stop=tmp_path/'pipeline/drain-label-worker.json';stop.parent.mkdir();stop.write_text('{}')
    async def exercise():
        started=asyncio.Event()
        class Provider:
            def __init__(self,cfg,**kwargs):
                assert cfg['require_airborne_verified'] and cfg['requests_per_minute']==120
            def request_drain(self):events.append('drain_requested')
            async def drain(self):await started.wait();events.append('paid_settled')
            async def close(self):events.append('closed')
        async def pending(*args,**kwargs):
            started.set()
            try:await asyncio.Event().wait()
            finally:events.append('label_cancelled')
        monkeypatch.setattr(providers,'GeminiProvider',Provider);monkeypatch.setattr(labelling,'label_frames',pending)
        monkeypatch.setattr(worker,'metadata_record_store',lambda *args,**kwargs:object())
        monkeypatch.setattr(label_archive,'archive_pass',lambda *args,**kwargs:{'status':'test'})
        await worker.run(path)
    asyncio.run(exercise())
    assert events.index('paid_settled')<events.index('label_cancelled')
    state=json.loads((tmp_path/'pipeline/label-worker.json').read_text());assert state['status']=='drained' and state['paid_calls_settled']
