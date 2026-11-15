import asyncio,hashlib,json
from types import SimpleNamespace
import pytest
from aerorozum.common import digest
from aerorozum.providers import request_identity
from aerorozum.evaluation_completions import completion_text


def test_request_identity_preserves_existing_cache_keys():
    cfg=dict(temperature=0.,max_retries=1)
    fields=dict(role='eval',prompt='prompt',images=['hash'],model='model',max_output_tokens=8192)
    expected=digest(dict(**fields,temperature=0.,api_version='v1beta',thinking_budget=None))
    assert request_identity(cfg,**fields)==expected


def test_malformed_paid_completion_is_reused_without_another_request(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'image');root=tmp_path/'provider';(root/'raw').mkdir(parents=True)
    cfg=dict(temperature=0.,max_retries=1);hashes=[hashlib.sha256(b'image').hexdigest()]
    key=request_identity(cfg,role='eval',prompt='prompt',images=hashes,model='model',max_output_tokens=8192)
    (root/'raw'/(key+'-0.json')).write_text(json.dumps(dict(id=key,role='eval',prompt='prompt',image_hashes=hashes,model='model',raw_text='{"detections":[')))
    async def generate(*a,**k):pytest.fail('Retained paid output must not be requested again')
    provider=SimpleNamespace(cfg=cfg,root=root,generate=generate)
    assert asyncio.run(completion_text(provider,'eval','prompt',[image],model='model',max_output_tokens=8192))=='{"detections":['


def test_transport_error_without_a_paid_completion_is_not_a_model_prediction(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'image')
    async def generate(*a,**k):raise RuntimeError('API unavailable')
    provider=SimpleNamespace(cfg=dict(temperature=0.,max_retries=1),root=tmp_path/'provider',generate=generate)
    with pytest.raises(RuntimeError,match='API unavailable'):
        asyncio.run(completion_text(provider,'eval','prompt',[image],model='model',max_output_tokens=8192))
