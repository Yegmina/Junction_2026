"""Mission-only client for the private VM Gemma endpoint."""
import base64
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from .common import save, utcnow


class LocalMissionProvider:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.url = os.environ.get('AEROROZUM_GEMMA_URL', 'http://127.0.0.1:18940').rstrip('/')
        parsed = urlsplit(self.url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
            raise ValueError('Gemma endpoint must use private loopback HTTP')

    async def generate(self, kind, prompt, images=(), model=None, max_output_tokens=768):
        request_id = uuid.uuid4().hex
        payload = dict(prompt=prompt, images=[base64.b64encode(Path(p).read_bytes()).decode() for p in images],
                       max_output_tokens=max_output_tokens)
        save(self.root/'requests'/(request_id+'.json'), dict(at=utcnow(), kind=kind, prompt=prompt,
             image_paths=[str(p) for p in images], model=model, backend='local_gemma'))
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(self.url+'/generate', json=payload)
            response.raise_for_status()
            result = response.json()
        save(self.root/'responses'/(request_id+'.json'), dict(at=utcnow(), **result))
        if not isinstance(result.get('output'), dict):
            raise ValueError('Local Gemma did not return valid JSON; raw output preserved')
        return result['output']
