"""Score the first paid completion, including invalid model formatting.

Strict provider/label generation remains unchanged. This evaluation-only path
reuses retained raw text and distinguishes a model syntax failure from a
transport failure with no completion. It requires a single-attempt protocol.
"""
import hashlib
import json
from pathlib import Path
from .providers import request_identity


async def completion_text(provider,role,prompt,images,*,model,max_output_tokens):
    if provider.cfg.get('max_retries')!=1:raise ValueError('First-completion evaluation requires exactly one provider attempt')
    hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images]
    identity=request_identity(provider.cfg,role=role,prompt=prompt,images=hashes,model=model,max_output_tokens=max_output_tokens)
    path=provider.root/'raw'/(identity+'-0.json')
    def retained():
        if not path.exists():return None
        row=json.loads(path.read_text())
        if any(row.get(k)!=v for k,v in dict(id=identity,role=role,prompt=prompt,image_hashes=hashes,model=model).items()):
            raise ValueError('Retained completion identity changed')
        return row.get('raw_text') or ''
    text=retained()
    if text is not None:return text
    try:output=await provider.generate(role,prompt,images,model=model,max_output_tokens=max_output_tokens)
    except RuntimeError:
        text=retained()
        if text is None:raise
        return text
    text=retained()
    return text if text is not None else json.dumps(output)
