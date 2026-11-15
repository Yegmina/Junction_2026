"""Read-only full-gradient checks with one small transfer per device.

Avoid a host/device synchronization for every LoRA tensor. Every gradient
element and the visual projection receive the same checks as the old monitor.
This changes observation overhead, not gradients or optimizer arithmetic.
"""
import math


def statistics(named_parameters):
    import torch
    groups={};count=0
    for name,parameter in named_parameters:
        grad=parameter.grad
        if not parameter.requires_grad or grad is None:continue
        count+=grad.numel()
        group=groups.setdefault(grad.device,dict(finite=[],squares=[],visual=[]))
        value=grad.detach()
        group['finite'].append(torch.isfinite(value).all())
        squared=value.float().square().sum()
        group['squares'].append(squared)
        if 'embed_vision.' in name and '.embedding_projection.' in name:
            group['visual'].append(squared)
    squared=0.;visual=0.
    for group in groups.values():
        # Float32 per-tensor sums match the original expression. Float64
        # aggregation replaces its Python float accumulation, with no writes
        # to any gradient tensor and no change to clipping or optimization.
        values=torch.stack(group['squares']).double()
        vis=torch.stack(group['visual']).double().sum() if group['visual'] else values.new_zeros(())
        packed=torch.stack((torch.stack(group['finite']).all().double(),values.sum(),vis)).cpu().tolist()
        if not packed[0]:raise FloatingPointError('Non-finite training gradient')
        squared+=packed[1];visual+=packed[2]
    if not count or not math.isfinite(squared):raise FloatingPointError('No finite LoRA gradients')
    return dict(count=count,squared=squared,visual_squared=visual)
