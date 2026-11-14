"""Private, single-GPU mission inference endpoint. Bind to loopback only."""
import asyncio
import base64
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


class Generate(BaseModel):
    prompt: str = Field(min_length=4, max_length=128000)
    images: list[str] = Field(default_factory=list, max_length=16)
    max_output_tokens: int = Field(default=768, ge=64, le=4096)


@asynccontextmanager
async def lifespan(app):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.9)
    path = os.environ['AEROROZUM_GEMMA_WEIGHTS']
    app.state.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=True)
    adapter = os.environ.get('AEROROZUM_GEMMA_ADAPTER')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    app.state.model = model
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    return {'ready': True, 'model': os.environ.get('AEROROZUM_MISSION_MODEL', 'google/gemma-4-12B-it'),
            'hosting': 'vm_gpu', 'adapter': bool(os.environ.get('AEROROZUM_GEMMA_ADAPTER'))}


def infer(request):
    import torch
    content = []
    for encoded in request.images:
        if len(encoded) > 4_000_000:
            raise ValueError('Image payload too large')
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 2_000_000:
                raise ValueError('Image dimensions too large')
            content.append({'type': 'image', 'image': image.convert('RGB')})
    content.append({'type': 'text', 'text': request.prompt})
    processor = app.state.processor
    inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
    inputs = {k: (v.to('cuda:0', dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to('cuda:0'))
              if torch.is_tensor(v) else v for k, v in inputs.items()}
    length = inputs['input_ids'].shape[-1]
    began = time.monotonic()
    with torch.inference_mode():
        generated = app.state.model.generate(**inputs, do_sample=False, num_beams=1,
            max_new_tokens=request.max_output_tokens, use_cache=True)
    text = processor.decode(generated[0, length:], skip_special_tokens=True)
    try:
        output = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip()))
    except ValueError:
        output = None
    return {'output': output, 'text': text, 'elapsed_s': time.monotonic() - began,
            'input_tokens': length, 'output_tokens': generated.shape[-1] - length,
            'model': os.environ.get('AEROROZUM_MISSION_MODEL', 'google/gemma-4-12B-it')}


@app.post('/generate')
async def generate(request: Generate):
    if app.state.lock.locked():
        raise HTTPException(429, 'A model request is already running')
    async with app.state.lock:
        try:
            return await asyncio.to_thread(infer, request)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
