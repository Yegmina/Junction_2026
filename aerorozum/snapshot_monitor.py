"""Read-only, predeclared adapter snapshots at fixed optimizer steps."""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from transformers import TrainerCallback
from .common import save,utcnow


def snapshot(model,path):
    path=Path(path)
    if path.exists():raise ValueError('A recorded adapter snapshot cannot be overwritten')
    cpu=torch.get_rng_state().clone();cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    python=random.getstate();numpy=np.random.get_state()
    versions=[(p,p._version,p.grad,p.grad._version if p.grad is not None else None) for p in model.parameters()]
    model.save_pretrained(path,safe_serialization=True)
    if not torch.equal(cpu,torch.get_rng_state()) or random.getstate()!=python:raise ValueError('Snapshot consumed a training RNG stream')
    current=np.random.get_state()
    if current[0]!=numpy[0] or not np.array_equal(current[1],numpy[1]) or current[2:]!=numpy[2:]:raise ValueError('Snapshot consumed the NumPy RNG stream')
    if cuda and any(not torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())):raise ValueError('Snapshot consumed CUDA randomness')
    if any(p._version!=v or p.grad is not g or g is not None and g._version!=gv for p,v,g,gv in versions):raise ValueError('Snapshot modified model parameters or gradients')
    return dict(at=utcnow(),adapter=str(path),rng_streams_unchanged=True,parameter_and_gradient_versions_unchanged=True)


class SnapshotMonitor(TrainerCallback):
    def __init__(self,output,steps=None):
        self.output=Path(output);self.steps=set(steps if steps is not None else json.loads(os.environ.get('AEROROZUM_SFT_SNAPSHOT_STEPS','[]')))
    def on_step_end(self,args,state,control,model=None,**kwargs):
        if state.global_step in self.steps:
            record=snapshot(model,self.output/('step-'+str(state.global_step))/'adapter')
            save(self.output/('step-'+str(state.global_step))/'receipt.json',dict(record,optimizer_step=state.global_step,actual_epoch=state.epoch))
