import asyncio
import base64
import json
from pathlib import Path
import httpx
import pytest
from aerorozum.local_mission_provider import LocalMissionProvider


@pytest.mark.parametrize('url', ['https://127.0.0.1:18940', 'http://example.com', 'http://user:password@localhost'])
def test_model_endpoint_must_be_private_loopback(tmp_path, monkeypatch, url):
    monkeypatch.setenv('AEROROZUM_GEMMA_URL', url)
    with pytest.raises(ValueError, match='loopback'):
        LocalMissionProvider(tmp_path)


def test_invalid_model_json_is_preserved_without_fallback(tmp_path, monkeypatch):
    image = tmp_path/'frame.png'
    image.write_bytes(b'image-bytes')
    original = httpx.AsyncClient
    async def handle(request):
        payload = json.loads(request.content)
        assert base64.b64decode(payload['images'][0]) == b'image-bytes'
        return httpx.Response(200, json={'output': None, 'text': 'invalid action', 'model': 'gemma-test'})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    provider = LocalMissionProvider(tmp_path/'provider')
    with pytest.raises(ValueError, match='raw output preserved'):
        asyncio.run(provider.generate('mission', 'Inspect east', images=[image]))
    responses = list((tmp_path/'provider/responses').glob('*.json'))
    assert len(responses) == 1
    assert json.loads(responses[0].read_text())['text'] == 'invalid action'
