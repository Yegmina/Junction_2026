"""Select training-only groups with real localization reward variation, then refine.

Invalid-vs-valid JSON variation alone never qualifies a group. All sampled
outputs and task rewards are retained, including rejected groups.
"""
import argparse,gc,json,os,random,time,hashlib
from pathlib import Path


def run(root,adapter,output,maximum=192):
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText,set_seed
    from peft import PeftModel
    from aerorozum.common import save,utcnow,digest
    from aerorozum.training import PROMPT,load_stream_manifest,dense_detection_reward,train
    from aerorozum.stream_training import evaluate_adapter,reward_signal_diagnostics
    root=Path(root);adapter=Path(adapter);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=root/'submission-refinement-v1/train.json';doc=json.loads(manifest.read_text())
    checked,_=load_stream_manifest(manifest,'stream_grpo',root=root)
    positives=[r for r in checked if r['gold_boxes']];random.Random(193).shuffle(positives)
    selected=[];source={r['sha256']:r for r in doc['images']};spec=dict(adapter=str(adapter),seed=17,groups=4,max_tokens=192,temperature=1.,maximum_training_images=maximum,sealed_test_accessed=False)
    save(output/'protocol.json',spec)
    torch.set_num_threads(2);set_seed(17);torch.cuda.set_per_process_memory_fraction(.88)
    model_path=os.environ['AEROROZUM_LOCAL_MODEL_PATH'];processor=AutoProcessor.from_pretrained(model_path)
    base=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval();model.config.use_cache=True
    for i,row in enumerate(positives[:maximum]):
        with Image.open(row['image_path']) as im:image=im.convert('RGB')
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
        inputs={k:(v.to('cuda',dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to('cuda')) if torch.is_tensor(v) else v for k,v in inputs.items()}
        prefix=inputs['input_ids'].shape[-1]
        with torch.inference_mode():generated=model.generate(**inputs,do_sample=True,temperature=1.,top_p=.95,num_return_sequences=4,max_new_tokens=192,use_cache=True)
        texts=[processor.decode(tokens[prefix:],skip_special_tokens=True) for tokens in generated]
        rewards=dense_detection_reward(texts,solution=[row['gold_boxes']]*len(texts));qualifies=max(rewards)>0 and float(np.std(rewards))>1e-6
        save(output/'raw'/(row['sha256']+'.json'),dict(image_sha256=row['sha256'],texts=texts,rewards=rewards,qualifies=qualifies))
        if qualifies:selected.append(source[row['sha256']])
        status=dict(id='submission-reward-curriculum-v1',kind='reward_supported_curriculum',status='running',completed=i+1,total=min(maximum,len(positives)),selected=len(selected),detail='Training-only localization reward-group discovery; invalid/valid formatting variation alone is insufficient',updated_at=utcnow())
        save(output/'status.json',status);save(root/'jobs/submission-reward-curriculum-v1.json',status)
        if len(selected)>=48:break
    del model,base,processor;gc.collect();torch.cuda.empty_cache()
    if len(selected)<8:raise ValueError('Fewer than eight training images have task-supported reward variation; do not run format-only RL')
    negatives=[r for r in doc['images'] if not r.get('boxes')];random.Random(197).shuffle(negatives)
    items=selected+negatives[:max(2,len(selected)//8)];random.Random(199).shuffle(items)
    identity=digest([spec,sorted(r['sha256'] for r in items)])[:24]
    frozen={**doc,'images':items,'metadata':{**doc['metadata'],'batch_id':identity,'experiment_branch':'reward_supported_curriculum_v1','reward_protocol':'soft-iou-f2-v2','selected_with_training_images_only':True}}
    batch=output/'manifest.json';save(batch,frozen);load_stream_manifest(batch,'stream_sft',root=root)
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE='submission-reward-curriculum-v1/'+identity,AEROROZUM_WARMSTART_ADAPTER=str(adapter),AEROROZUM_SFT_LEARNING_RATE='2e-6',AEROROZUM_DEVICE_MAP='balanced')
    sft=train(batch,'stream_sft',17,16);gc.collect();torch.cuda.empty_cache()
    os.environ['AEROROZUM_SFT_CHECKPOINT']=str(root/sft['adapter'])
    grpo=train(batch,'stream_grpo',17,64);gc.collect();torch.cuda.empty_cache()
    evaluation={}
    for name,result in [('sft',sft),('grpo',grpo)]:
        evaluation[name]=evaluate_adapter(root/result['adapter'],root/'pipeline/validation.json',output/'evaluation'/name,model_path,seed=17)
        gc.collect();torch.cuda.empty_cache()
    result=dict(status='completed',selected_images=len(selected),sft=sft,grpo=grpo,evaluation=evaluation,reward_signal=reward_signal_diagnostics((root/grpo['adapter']).parent/'metrics.jsonl'),frontier_superiority_verified=False)
    save(output/'result.json',result);status.update(status='completed',detail='Curriculum, matched SFT/GRPO and development scoring completed',updated_at=utcnow());save(output/'status.json',status);save(root/'jobs/submission-reward-curriculum-v1.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:run(a.root,a.adapter,a.output)
    except Exception as error:
        from aerorozum.common import save,utcnow
        save(Path(a.output)/'status.json',dict(status='failed',error=str(error),at=utcnow()));raise
