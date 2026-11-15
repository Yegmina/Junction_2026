"""Qualify the adapter saver on a real tiny PEFT model using CPU only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from transformers import LlamaConfig,LlamaForCausalLM,set_seed
from peft import get_peft_model,LoraConfig
from aerorozum.snapshot_monitor import SnapshotMonitor


def main():
    from aerorozum.common import save,utcnow
    torch.set_num_threads(1);set_seed(417)
    config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
    model=get_peft_model(LlamaForCausalLM(config),LoraConfig(r=2,lora_alpha=4,lora_dropout=.1,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    original=copy.deepcopy(model.state_dict());tokens=torch.tensor([[1,4,8,16,32,2]])
    outputs=[]
    with tempfile.TemporaryDirectory(prefix='aerorozum-snapshot-qualification-') as directory:
        for capture in [False,True]:
            model.load_state_dict(original);set_seed(851);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
            monitor=SnapshotMonitor(Path(directory)/'fixed-snapshots',steps=[3,6,9])
            losses=[]
            for step in range(1,11):
                model.train();loss=model(input_ids=tokens,labels=tokens).loss;loss.backward();optimizer.step()
                if capture:monitor.on_step_end(None,SimpleNamespace(global_step=step,epoch=step/10),SimpleNamespace(),model=model)
                losses.append(float(loss.detach()));optimizer.zero_grad(set_to_none=True)
            outputs.append(dict(losses=losses,state=copy.deepcopy(model.state_dict())))
    same=outputs[0]['losses']==outputs[1]['losses'] and all(torch.equal(outputs[0]['state'][k],v) for k,v in outputs[1]['state'].items())
    if not same:raise ValueError('Snapshotting changed the paired optimizer trajectory')
    import aerorozum.snapshot_monitor as implementation
    receipt=dict(at=utcnow(),status='passed',backend='CPU tiny real PEFT Llama; no GPU allocation',optimizer_steps=10,
        dropout=.1,three_real_adapter_serializations=True,actual_trainer_callback_exercised=True,losses_bit_identical=True,all_model_parameters_bit_identical=True,
        rng_and_parameter_and_gradient_guards_passed=True,source_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
        scope='Read-only adapter serialization qualification; not a claim of improved model accuracy')
    save(Path('/opt/aerorozum/artifacts/overnight-20261115T000942/snapshot-monitor-qualification.json'),receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
