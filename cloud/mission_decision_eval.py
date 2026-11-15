"""Isolated offline GPU evaluation. No API credentials, flight or training calls."""
import argparse,hashlib,json,os,re,subprocess,threading,time
from pathlib import Path
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score(case,text):
    try:
        value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()));assert isinstance(value,dict)
    except Exception:return dict(json_valid=False,action_valid=False,contract_success=False,hallucinated_without_image=False)
    action=value.get('action');state=case['public_state'];expected=case['expected'];valid=action in ('inspect_area','revisit','return_home')
    if action in ('inspect_area','revisit'):valid=valid and value.get('area_id') in state['allowed_areas']
    success=valid and action==expected['action']
    if expected.get('allowed_area_ids'):success=success and value.get('area_id') in expected['allowed_area_ids']
    if 'max_altitude_m' in expected:
        altitude=value.get('altitude_m');success=success and isinstance(altitude,(int,float)) and not isinstance(altitude,bool) and 2<=altitude<=expected['max_altitude_m']
    hallucinated=not case['images'] and bool(value.get('detections'))
    if expected.get('no_detections'):success=success and not hallucinated
    return dict(json_valid=True,action_valid=bool(valid),contract_success=bool(success),hallucinated_without_image=bool(hallucinated))

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',required=True);p.add_argument('--adapter');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--gcs',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);corpus_path=Path(a.corpus);doc=json.loads(corpus_path.read_text());cases=sorted(doc['cases'],key=lambda c:len(c['images']))
    status=dict(status='loading_model',variant=a.variant,completed=0,total=len(cases),at=now(),gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),scope=doc['scope']);save(root/'status.json',status)
    protocol=dict(corpus_sha256=doc['corpus_sha256'],variant=a.variant,model='google/gemma-4-12B-it',adapter=a.adapter,decoding=doc['decoding'],gold_visible_to_model=False,scope=doc['scope'],checkpoint_comparison=doc['checkpoint_comparison'],batch_size_requested=a.batch_size,started_at=now())
    if a.adapter:
        adapter=Path(a.adapter);protocol['adapter_sha256']=sha(adapter/'adapter_model.safetensors');protocol['adapter_config_sha256']=sha(adapter/'adapter_config.json');protocol['training_summary']=json.loads((adapter.parent/'summary.json').read_text())
    save(root/'protocol.json',protocol)
    def sync():
        try:subprocess.run(['gcloud','storage','rsync','-r',str(root),a.gcs,'--quiet'],capture_output=True,timeout=80,check=True)
        except Exception as e:save(root/'archive-error.json',dict(at=now(),error=str(e)[:400]))
    stop=threading.Event()
    def uploader():
        sync()
        while not stop.wait(40):sync()
    thread=threading.Thread(target=uploader,daemon=True);thread.start()
    try:
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText,set_seed
        from PIL import Image
        set_seed(20261114);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.9)
        model_path='/opt/aerorozum/models/gemma-4-12B-it';processor=AutoProcessor.from_pretrained(model_path,local_files_only=True);processor.tokenizer.padding_side='left'
        model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True,local_files_only=True)
        if a.adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
        model.eval();model.config.use_cache=True;results=[];started=time.monotonic();index=0;batch_size=a.batch_size
        while index<len(cases):
            batch=cases[index:index+batch_size]
            # Keep image count uniform within a processor batch.
            count=len(batch[0]['images']);batch=[c for c in batch if len(c['images'])==count]
            messages=[]
            for c in batch:
                content=[]
                for image in c['images']:
                    path=corpus_path.parent/image['path']
                    if sha(path)!=image['sha256']:raise ValueError('Frozen input checksum mismatch')
                    with Image.open(path) as source:content.append(dict(type='image',image=source.convert('RGB')))
                content.append(dict(type='text',text=c['prompt']));messages.append([dict(role='user',content=content)])
            try:
                inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',padding=True)
                inputs={k:(v.to(device='cuda:0',dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to('cuda:0')) if torch.is_tensor(v) else v for k,v in inputs.items()}
                token_count=inputs['input_ids'].shape[-1];began=time.monotonic()
                with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=320,use_cache=True)
            except torch.cuda.OutOfMemoryError:
                if batch_size==1:raise
                batch_size=max(1,batch_size//2);torch.cuda.empty_cache();save(root/'batch-adjustment.json',dict(batch_size=batch_size,reason='OOM: preserve model precision; reduce batch only',at=now()));continue
            ids=generated[:,token_count:].detach().cpu().tolist();texts=processor.batch_decode(ids,skip_special_tokens=True)
            for c,text,tokens in zip(batch,texts,ids):
                raw=dict(case_id=c['id'],variant=a.variant,corpus_sha256=doc['corpus_sha256'],text=text,generated_token_ids=tokens,batch_size_actual=len(batch),elapsed_batch_s=time.monotonic()-began,input_tokens=token_count,input_token_sha256=hashlib.sha256(inputs['input_ids'][len(results)%len(batch)].detach().cpu().numpy().tobytes()).hexdigest(),at=now())
                save(root/'raw'/(c['id']+'.json'),raw)
                result=dict(case_id=c['id'],category=c['category'],**score(c,text));save(root/'scores'/(c['id']+'.json'),result);results.append(result)
            index+=len(batch);status.update(status='running',completed=index,elapsed_s=time.monotonic()-started,gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3,at=now());save(root/'status.json',status);print(json.dumps(status),flush=True)
        metrics={k:sum(r[k] for r in results)/len(results) for k in ('json_valid','action_valid','contract_success','hallucinated_without_image')}
        groups={key:dict(cases=sum(r['category']==key for r in results),contract_success_rate=sum(r['contract_success'] for r in results if r['category']==key)/sum(r['category']==key for r in results)) for key in sorted({r['category'] for r in results})}
        save(root/'summary.json',dict(status='completed',variant=a.variant,cases=len(results),metrics=metrics,by_category=groups,scope=doc['scope'],protocol=protocol,at=now()))
        status.update(status='completed',metrics=metrics,at=now());save(root/'status.json',status)
    except Exception as e:
        status.update(status='failed',error=type(e).__name__+': '+str(e)[:1000],at=now());save(root/'status.json',status);raise
    finally:stop.set();thread.join(timeout=85);sync()

if __name__=='__main__':main()
