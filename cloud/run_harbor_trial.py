"""Run a pinned OpenCode model through Harbor without putting API keys in argv.

The Google API key stays in the process environment/private credential file.
Local Gemma uses a dummy credential and the private OpenAI-compatible endpoint.
"""
import argparse,json,os,shutil,subprocess,tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--job-name',required=True);p.add_argument('--jobs-dir',default='./jobs');p.add_argument('--harbor',default=shutil.which('harbor') or 'harbor');p.add_argument('--model',default='gemini-4-argon');p.add_argument('--local-gemma',action='store_true');p.add_argument('--local-model',default='gemma-4-12b-refined-grpo');p.add_argument('--base-url',default='http://host.docker.internal:18941/v1');p.add_argument('--seed',type=int);p.add_argument('--key-file');p.add_argument('--phase',choices=['warming','cooling','dawn','late_warming'],default='warming');p.add_argument('--budget',type=int,default=180);a=p.parse_args()
    env=os.environ.copy()
    if not 60<=a.budget<=600:p.error('--budget must be 60..600 simulated seconds')
    env.update(AEROROZUM_TASK_PHASE=a.phase,AEROROZUM_TASK_BUDGET=str(a.budget))
    if a.seed is not None:env['AEROROZUM_TASK_SEED']=str(a.seed)
    extra=[]
    if a.local_gemma:
        model=a.local_model;provider='aerorozum';selector=provider+'/'+model
        env['OPENAI_API_KEY']='local-inference-only'
        extra=['--base-url',a.base_url,'--api-key-name','OPENAI_API_KEY','--api-format','openai_chat_completions']
        key_name='OPENAI_API_KEY';context=24576;output=4096
    else:
        model=a.model;provider='google';selector='gemini/'+model
        key=os.environ.get('GOOGLE_GENERATIVE_AI_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not key and a.key_file:key=Path(a.key_file).read_text().strip()
        if not key:raise RuntimeError('Set GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY, or provide --key-file')
        env['GOOGLE_GENERATIVE_AI_API_KEY']=key
        extra=['--api-key-name','GOOGLE_GENERATIVE_AI_API_KEY'];key_name='GOOGLE_GENERATIVE_AI_API_KEY';context=131072;output=8192
    config=dict(provider={provider:dict(options=dict(apiKey='{env:'+key_name+'}'),models={model:dict(name=model,limit=dict(context=context,output=output),modalities=dict(input=['text','image'],output=['text']))})})
    kwargs=dict(version='1.19.5',opencode_config=config)
    if a.local_gemma:
        config['provider'][provider].update(npm='@ai-sdk/openai-compatible')
        config['provider'][provider]['options']['baseURL']=a.base_url
        job=dict(schema_version=1,job_name=a.job_name,jobs_dir=a.jobs_dir,n_concurrent_trials=1,
            agents=[dict(name='opencode',model_name=selector,skip_model_parsing=True,
                base_url=a.base_url,model_api_key_name='OPENAI_API_KEY',
                model_api_format='openai_chat_completions',kwargs=kwargs)],tasks=[dict(path=a.task)])
        with tempfile.TemporaryDirectory(prefix='harbor-local-provider-') as temp:
            path=Path(temp)/'job.json';path.write_text(json.dumps(job))
            code=subprocess.run([a.harbor,'run','-c',str(path)],env=env).returncode
    else:
        args=[a.harbor,'run','-p',a.task,'-a','opencode','-m',selector,*extra,'--ak','version=1.19.5','--ak','opencode_config='+json.dumps(config),'-o',a.jobs_dir,'--job-name',a.job_name,'-n','1']
        code=subprocess.run(args,env=env).returncode
    # A paired comparison is invalid if the actual instance differs from its
    # requested seed, even when a run otherwise produced a positive reward.
    if a.seed is not None:
        ledgers=list((Path(a.jobs_dir)/a.job_name).glob('*/verifier/ledger.json'))
        identities=[json.loads(path.read_text()).get('instance',{}).get('seed') for path in ledgers]
        receipt=dict(requested_seed=a.seed,actual_seeds=identities,matched=bool(identities) and all(seed==a.seed for seed in identities))
        destination=Path(a.jobs_dir)/a.job_name/'instance-check.json';destination.write_text(json.dumps(receipt,indent=2))
        if not receipt['matched']:code=1
    return code


if __name__=='__main__':raise SystemExit(main())
