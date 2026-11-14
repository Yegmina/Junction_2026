"""Full-BF16 single-GPU backward and GRPO memory probes; no quantization."""
import argparse,gc,json,os,time
from pathlib import Path
from .common import data_root,save,utcnow

def probe(batch,output,seed=17):
    import torch
    from .training import train
    if torch.cuda.device_count()!=1:raise ValueError('Probe requires one visible GPU')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);batch_id=json.loads(Path(batch).read_text())['metadata']['batch_id']
    total=torch.cuda.get_device_properties(0).total_memory;trials=[]
    os.environ.pop('AEROROZUM_DEVICE_MAP',None);os.environ.pop('AEROROZUM_WARMSTART_ADAPTER',None)
    os.environ.update(AEROROZUM_MEMORY_PROBE='1',AEROROZUM_GPU_MEMORY_FRACTION='.88');torch.set_num_threads(2)
    for micro,accum in ((1,4),(2,2)):
        namespace=f'gpu-memory-probe/{batch_id}/micro-{micro}'
        os.environ.update(AEROROZUM_TRAINING_NAMESPACE=namespace,AEROROZUM_MICROBATCH=str(micro),AEROROZUM_GRAD_ACCUMULATION=str(accum))
        started=time.monotonic();trial={'microbatch':micro,'gradient_accumulation':accum,'effective_batch':4,'quantization':False,'dtype':'bfloat16'}
        try:
            sft=train(batch,'stream_sft',seed,1);gc.collect();torch.cuda.empty_cache()
            os.environ['AEROROZUM_SFT_CHECKPOINT']=str(data_root()/sft['adapter']);grpo=train(batch,'stream_grpo',seed,1)
            peak=max(sft['peak_gpu_memory_bytes'],grpo['peak_gpu_memory_bytes'],sft.get('peak_gpu_reserved_bytes',0),grpo.get('peak_gpu_reserved_bytes',0))
            trial.update(status='passed' if peak<=.9*total else 'above_memory_limit',peak_memory_bytes=peak,memory_fraction=peak/total,
                forward_backward_verified=sft['nonzero_gradient_checks']>0,grpo_verified=grpo['nonzero_gradient_checks']>0)
        except Exception as exc:trial.update(status='failed',error=str(exc)[:1500])
        finally:gc.collect();torch.cuda.empty_cache()
        trial['elapsed_s']=time.monotonic()-started;trials.append(trial);save(output/'progress.json',{'trials':trials,'at':utcnow()})
        if trial['status']!='passed':break
    passed=[r for r in trials if r['status']=='passed' and r.get('forward_backward_verified') and r.get('grpo_verified')]
    selected=passed[-1] if passed else None
    if len(passed)==2 and passed[1]['elapsed_s']>passed[0]['elapsed_s']*1.1:selected=passed[0]
    result={'status':'passed' if selected else 'failed','single_gpu':True,'gpu_total_memory_bytes':total,
        'visible_gpu_ids':os.environ.get('CUDA_VISIBLE_DEVICES'),'trials':trials,'selected':selected,'at':utcnow(),
        'scope':'Measured memory compatibility; full research schedule unchanged'}
    save(output/'result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=17);a=p.parse_args()
    try:print(json.dumps(probe(a.batch,a.output,a.seed)),flush=True)
    except Exception as exc:save(Path(a.output)/'result.json',{'status':'failed','error':str(exc)[:2000],'at':utcnow()});raise

if __name__=='__main__':main()
