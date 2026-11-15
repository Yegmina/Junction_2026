"""Bounded cloud-only equivalence and synchronization-overhead benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time


def original(named):
    import torch
    squared=0.;visual=0.;count=0
    for name,p in named:
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():raise FloatingPointError('Non-finite training gradient')
            value=float(p.grad.float().square().sum());squared+=value;count+=p.grad.numel()
            if 'embed_vision.' in name and '.embedding_projection.' in name:visual+=value
    if not count or not math.isfinite(squared):raise FloatingPointError('No finite LoRA gradients')
    return dict(count=count,squared=squared,visual_squared=visual)


def run(output):
    import torch
    from aerorozum.gradient_monitor import statistics
    from aerorozum.common import save,utcnow
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.04);torch.manual_seed(20261115)
    named=[]
    for i in range(128):
        name=('embed_vision.tower.embedding_projection.' if i%8==0 else 'language.layers.')+str(i)
        named.append((name,SimpleNamespace(requires_grad=True,grad=torch.randn((16,6144),device='cuda',dtype=torch.bfloat16))))
    old=original(named);new=statistics(named)
    equivalent=old['count']==new['count'] and all(math.isclose(old[k],new[k],rel_tol=1e-12,abs_tol=1e-12) for k in ['squared','visual_squared'])
    if not equivalent:raise ValueError('Gradient monitor statistics differ')
    copies=[p.grad.clone() for _,p in named]
    for value in [float('nan'),float('inf'),-float('inf')]:
        before=named[0][1].grad[0,0].clone();named[0][1].grad[0,0]=value
        for function in [original,statistics]:
            try:function(named)
            except FloatingPointError:pass
            else:raise ValueError('A non-finite gradient was missed')
        named[0][1].grad[0,0]=before
    if not all(torch.equal(p.grad,c) for (_,p),c in zip(named,copies)):
        raise ValueError('A monitor changed a gradient')
    timing={}
    for name,function in [('original',original),('device_aggregated',statistics)]:
        values=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();function(named);torch.cuda.synchronize();values.append(time.perf_counter()-start)
        timing[name]=dict(samples_s=values,mean_s=sum(values)/len(values))
    # Check optimizer trajectories are bit-identical with and without the
    # read-only monitor, independently of the norm reporting tolerance.
    a=torch.nn.Parameter(torch.randn((16,64),device='cuda'));b=torch.nn.Parameter(a.detach().clone())
    oa=torch.optim.AdamW([a],lr=1e-4);ob=torch.optim.AdamW([b],lr=1e-4)
    for _ in range(10):
        grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
        original([('a',a)]);statistics([('b',b)]);oa.step();ob.step()
    if not torch.equal(a,b):raise ValueError('Optimizer trajectories changed')
    receipt=dict(at=utcnow(),status='passed',equivalent_statistics=equivalent,
        unchanged_gradients=True,identical_optimizer_trajectory=True,nan_and_infinity_detected=True,
        tensor_count=128,shape=[16,6144],dtype='bfloat16',timing=timing,
        monitor_only_speedup=timing['original']['mean_s']/timing['device_aggregated']['mean_s'],
        end_to_end_training_speedup_measured=False,actual_training_quality_change_claimed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        monitor_source_sha256=hashlib.sha256(Path(__import__('aerorozum.gradient_monitor',fromlist=['x']).__file__).read_bytes()).hexdigest())
    save(output,receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
