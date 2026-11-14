"""Measure the frozen base control, then continue the selected long-run adapter."""
import gc
import json
import os
from pathlib import Path


def main():
    import torch
    from aerorozum.common import data_root,save
    from aerorozum.stream_training import cached_baseline
    from refine_perception import run
    root=data_root();output=root/'submission-refinement-v3';output.mkdir(exist_ok=True)
    baseline=cached_baseline(None,root/'pipeline/validation.json',output/'base-control',os.environ['AEROROZUM_LOCAL_MODEL_PATH'],seed=17)
    save(output/'base-control/result.json',baseline);gc.collect();torch.cuda.empty_cache()
    spec=dict(namespace='submission-refinement-v3/updates-480',manifest=str(root/'submission-refinement-v1/train.json'),validation=str(root/'pipeline/validation.json'),output=str(output/'updates-480'),learning_rate=1e-5,sft_steps=480,grpo_steps=64,
        warm_start=str(root/'training/submission-refinement-v2/updates-480/google_gemma-4-12B-it/stream_grpo/seed-17/adapter'))
    save(output/'specification.json',spec);run(spec)


if __name__=='__main__':main()
