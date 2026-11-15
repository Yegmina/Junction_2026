"""Require completed SFT and task-learning lineage for GRPO-only continuation."""
import hashlib
import json
from pathlib import Path


def validate_continuation(result_path,*,root,model,seed,manifest_hash,lora_scope):
    from .training import validate_parent_adapter
    root=Path(root).resolve();result_path=Path(result_path).resolve()
    if not result_path.is_relative_to(root):raise ValueError('Continuation result escapes artifact root')
    raw=result_path.read_bytes();document=json.loads(raw)
    if document.get('status')!='completed':raise ValueError('Continuation requires a completed parent experiment')
    result=document.get('results',{})
    if result.get('reward_signal',{}).get('task_signal_verified') is not True:
        raise ValueError('Parent GRPO has no verified task-reward variation')
    receipt=dict(parent_result=str(result_path.relative_to(root)),parent_result_sha256=hashlib.sha256(raw).hexdigest(),
        optimizer_state='Fresh optimizer; continue adapter weights, not an optimizer checkpoint')
    for name,stage in [('sft','stream_sft'),('grpo','stream_grpo')]:
        recorded=result.get(name,{})
        adapter=(root/recorded.get('adapter','')).resolve()
        if not adapter.is_relative_to(root/'training'):raise ValueError('Parent adapter escapes training tree')
        summary=validate_parent_adapter(adapter,stage,model,seed,manifest_hash)
        if summary.get('lora_scope','language')!=lora_scope:raise ValueError('Continuation adapter scope changed')
        if summary.get('optimizer_updates',0)<1 or summary.get('nonzero_gradient_checks',0)<1:
            raise ValueError('Parent has no verified optimizer updates')
        if any(recorded.get(k)!=summary.get(k) for k in ['model','seed','stage','manifest_hash','optimizer_updates','adapter']):
            raise ValueError('Parent experiment and completion summary disagree')
        receipt[name+'_adapter_sha256']=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
        if name=='grpo':parent=adapter
    return parent,receipt
