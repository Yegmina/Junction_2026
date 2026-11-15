"""Versioned additive coordinator: accepted arrivals -> shards -> frozen GPU passes.

Full research counts are preserved; incremental passes are a separate namespace.
Only an external physics acceptance report can admit synthetic observations.
"""
from __future__ import annotations
import argparse,copy,json,math,os,random,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import fcntl
from .common import config as load_config,data_root,digest,save,utcnow
from .fast_fields import file_hash,render_pair_shard

DEFAULT={'namespace':'incremental-stream-v1','minimum_native_positive':32,'minimum_native_zero':32,
    'minimum_new_rows':64,'maximum_new_rows':256,'manual_anchors':64,'replay_rows':64,
    'synthetic_shard_images':64,'sft_steps':32,'grpo_steps':16,'seeds':[17,43,91],
    'gpu_groups':[[0,1],[2,3]],'maximum_queued_jobs':6,'poll_seconds':15,
    'deadline_utc':'2026-11-15T06:30:50+00:00','train_threads':2,'maximum_synthetic_fraction':.25}


class StreamCoordinator:
    def __init__(self,root=None,settings=None,executor=None):
        self.artifact_root=Path(root or data_root()).resolve();self.root=self.artifact_root/'pipeline';self.root.mkdir(parents=True,exist_ok=True)
        self.settings={**DEFAULT,**(settings or {})};self.executor=executor;self.processes={};self.row_cache={}
        if not 0<=self.settings['maximum_synthetic_fraction']<=.25:raise ValueError('Auxiliary physical crops cannot exceed25% of an incremental batch')
        self.state_path=self.root/'coordinator.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'schema_version':1,'namespace':DEFAULT['namespace'],'accepted_pairs':{},'rejected_pairs':{},'consumed':[],
            'batches':[],'jobs':[],'promoted':{},'created_at':utcnow()}
        if self.state['namespace']!=self.settings['namespace']:raise ValueError('Incremental namespace mismatch')
        self.state.setdefault('evaluation_jobs',[])
        self.state.setdefault('job_requests',{})

    def _save(self):save(self.state_path,self.state)

    def _read(self,path,default):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):return copy.deepcopy(default)

    def _eligible(self,row,sessions,allowed_split='train'):
        if row.get('split')!=allowed_split:return False
        if (allowed_split=='train' and row.get('training_eligible') is False) or row.get('cross_split_duplicate') is True:return False
        zone=row.get('zone') or row.get('parent_zone') or row.get('source',{}).get('zone')
        if zone in ('zone8','zone9') or (allowed_split=='train' and zone=='zone7'):return False
        image=Path(row.get('image',''))
        if image.is_absolute() or '..' in image.parts:return False
        path=(self.artifact_root/image).resolve()
        if not path.is_relative_to(self.artifact_root) or not path.is_file():return False
        sha=row.get('sha256');fingerprint=(str(path),path.stat().st_mtime_ns,path.stat().st_size)
        if fingerprint not in self.row_cache:self.row_cache[fingerprint]=file_hash(path)
        if sha!=self.row_cache[fingerprint]:return False
        provenance=row.get('provenance');source=row.get('source',{})
        if provenance=='user_accepted_model':
            if row.get('model')!='gemini-robotics-er-2-preview' or row.get('user_threshold')!=.6 or row.get('human_reviewed') is not False:return False
            if source.get('session_id') and sessions.get(source['session_id'])!=allowed_split:return False
            detections=row.get('detections')
            if not isinstance(detections,list) or row.get('uncertain_boxes') or row.get('training_excluded_reason'):return False
            if any(not isinstance(d.get('confidence'),(int,float)) or d['confidence']<.6 for d in detections):return False
            if len(detections)!=len(row.get('boxes',[])):return False
            if not detections and row.get('zero_candidates') is not True:return False
        elif provenance in ('original_manual','reviewed_ai'):
            if zone not in ('zone1','zone2','zone3','zone4','zone5','zone7'):return False
        elif provenance in ('direct_physics','validated_surrogate'):
            if row.get('research_accepted') is not True or not row.get('acceptance_certificate_sha256'):return False
        else:return False
        from .evaluation import valid_box
        return all(valid_box(b.get('bbox',b)) for b in row.get('boxes',[]))

    def ingest_pairs(self):
        registry=self._read(self.settings.get('physics_registry','/opt/aerorozum/physics/registry.json'),{'fields':[]})
        for entry in registry.get('fields',[]):
            key=str(entry.get('campaign_id') or entry.get('field_path'));marker=Path(entry.get('acceptance_report_path',''))
            if not marker.is_file():continue
            revision=digest([key,file_hash(marker)])
            if revision in self.state['accepted_pairs'] or revision in self.state['rejected_pairs']:continue
            try:
                doc=render_pair_shard(entry,self.root/'synthetic',self.artifact_root,
                    count=self.settings['synthetic_shard_images'],seed=17)
                manifest=next((self.root/'synthetic').glob('*/manifest.json')) if not doc['images'] else self.artifact_root/Path(doc['images'][0]['image']).parents[1]/'manifest.json'
                self.state['accepted_pairs'][revision]={'campaign_id':key,'manifest':str(manifest),
                    'acceptance_report_path':str(marker),'images':len(doc['images']),'at':utcnow()}
            except Exception as exc:self.state['rejected_pairs'][revision]={'campaign_id':key,'reason':str(exc)[:1000],'at':utcnow()}
            self._save()
            # Bounded CPU/disk work: at most one newly arrived pair per tick.
            break

    def _catalog(self):
        real=self._read(self.root/'inputs/real.json',{'images':[],'session_splits':{}});sessions=real.get('session_splits',{})
        sessions=sessions.get('sessions',sessions);all_rows=list(real.get('images',[]));certificates={}
        for pair in self.state['accepted_pairs'].values():
            doc=self._read(pair['manifest'],{'images':[]});all_rows.extend(doc['images'])
            for row in doc['images']:certificates[row['acceptance_certificate_sha256']]=pair['acceptance_report_path']
        held_hashes={r.get('sha256') for r in all_rows if r.get('split') in ('validation','test','fast_test','direct_test')}
        held_hashes.update(real.get('sealed_test_image_hashes',[]))
        physical_keys=('physical_field_sha256','counterfactual_sha256','paired_present_sha256','paired_absent_sha256')
        held_physics={r.get(key) for r in all_rows if r.get('split') in ('validation','test','fast_test','direct_test') for key in physical_keys if r.get(key)}
        unique={}
        for row in all_rows:
            if row.get('sha256') in held_hashes and row.get('split')=='train':continue
            if row.get('split')=='train' and any(row.get(key) in held_physics for key in physical_keys if row.get(key)):continue
            if self._eligible(row,sessions,row.get('split')) and row.get('split') in ('train','validation'):
                unique.setdefault(row['sha256'],row)
        return list(unique.values()),sessions,certificates,real

    def freeze_arrivals(self):
        rows,sessions,certificates,source=self._catalog();train=[r for r in rows if r['split']=='train']
        validation=self.root/'validation.json'
        if not validation.exists():
            manual=sorted([r for r in rows if r['split']=='validation' and r.get('zone')=='zone7'],key=lambda r:r['id'])
            native=sorted([r for r in rows if r['split']=='validation' and r.get('provenance')=='user_accepted_model'],key=lambda r:r['id'])[:32]
            if len(manual)>=8:save(validation,{'metadata':{'frozen_development':True,'created_at':utcnow(),'test_zones_sealed':['zone8','zone9']},'images':manual+native})
        pending=sum(job['status'] in ('queued','running') for job in self.state['jobs'])
        consumed=set(self.state['consumed']);new=[r for r in train if r['sha256'] not in consumed and r['provenance'] not in ('original_manual','reviewed_ai')]
        positive=[r for r in new if r.get('boxes')];negative=[r for r in new if not r.get('boxes')]
        native_positive=[r for r in positive if r['provenance']=='user_accepted_model'];native_zero=[r for r in negative if r['provenance']=='user_accepted_model']
        synthetic_positive=[r for r in positive if r['provenance']=='direct_physics']
        native_ready=len(native_positive)>=self.settings['minimum_native_positive'] and len(native_zero)>=self.settings['minimum_native_zero']
        synthetic_ready=len(synthetic_positive)>=8
        self.state['arrival_gate']={'unseen_native_positive':len(native_positive),'unseen_explicit_zero':len(native_zero),
            'required_native_positive':self.settings['minimum_native_positive'],'required_explicit_zero':self.settings['minimum_native_zero'],
            'native_ready':native_ready,'synthetic_ready':synthetic_ready,'pending_gpu_jobs':pending,
            'reason':None if native_ready or synthetic_ready else f"Need {max(0,self.settings['minimum_native_positive']-len(native_positive))} more unseen train positives; negative-only repetition is excluded"}
        if pending+len(self.settings['seeds'])>self.settings['maximum_queued_jobs'] or not validation.exists():return
        if not native_ready and not synthetic_ready:return
        anchors=[r for r in train if r['provenance'] in ('original_manual','reviewed_ai') and r.get('boxes')]
        if len(anchors)<16:return
        rng=random.Random(17+len(self.state['batches']));rng.shuffle(positive);rng.shuffle(negative);rng.shuffle(anchors)
        # Source-group round robin reduces adjacent-frame/background dominance.
        def diverse(items,limit):
            groups={}
            for row in items:
                key=row.get('source_group') or row.get('source',{}).get('session_id') or row.get('zone') or 'unknown'
                groups.setdefault(key,[]).append(row)
            result=[]
            while groups and len(result)<limit:
                for key in sorted(list(groups)):
                    if groups[key] and len(result)<limit:result.append(groups[key].pop())
                    if not groups[key]:groups.pop(key)
            return result
        max_new=self.settings['maximum_new_rows'];selected_positive=diverse(positive,max_new//2)
        selected_negative=diverse(negative,min(max_new-len(selected_positive),2*len(selected_positive)))
        new_rows=selected_positive+selected_negative
        if len(new_rows)<self.settings['minimum_new_rows'] and not synthetic_ready:return
        replay=[r for r in train if r['sha256'] in consumed and r['provenance'] not in ('original_manual','reviewed_ai')]
        rng.shuffle(replay);selected=new_rows+anchors[:self.settings['manual_anchors']]+replay[:self.settings['replay_rows']]
        # These true physical patches lack calibrated recorded appearance. Keep
        # them auxiliary; realistic native/manual pixels remain the majority.
        real_selected=[r for r in selected if r['provenance'] not in ('direct_physics','validated_surrogate')]
        simulated=[r for r in selected if r['provenance'] in ('direct_physics','validated_surrogate')]
        fraction=self.settings['maximum_synthetic_fraction'];limit=int(len(real_selected)*fraction/(1-fraction))
        simulated=diverse([r for r in simulated if r.get('boxes')],(limit+1)//2)+diverse([r for r in simulated if not r.get('boxes')],limit//2)
        selected=real_selected+simulated;selected_hashes={r['sha256'] for r in selected}
        new_rows=[r for r in new_rows if r['sha256'] in selected_hashes]
        if not new_rows:return
        unique={r['sha256']:r for r in selected};selected=list(unique.values());rng.shuffle(selected)
        batch_id=digest({'namespace':self.settings['namespace'],'index':len(self.state['batches']),
                         'hashes':sorted(unique),'new_train_hashes':sorted(r['sha256'] for r in new_rows)})[:24]
        path=self.root/'batches'/batch_id/'manifest.json'
        doc={'metadata':{'frozen_stream_batch':True,'namespace':self.settings['namespace'],'batch_id':batch_id,'status':'passed','created_at':utcnow(),
            'new_images':len(new_rows),'manual_anchors':min(len(anchors),self.settings['manual_anchors']),
            'maximum_synthetic_fraction':fraction,'synthetic_training_role':'auxiliary_physics_cue_only','recorded_background_hybrid_training':'blocked until independently validated',
            'reward_protocol':'soft-iou-f2-v2',
            'accepted_pair_certificates':certificates,'full_research_protocol_unchanged':True},'session_splits':sessions,'images':selected}
        if path.exists():
            old=json.loads(path.read_text())
            if old['metadata'].get('batch_id')!=batch_id or sorted(r['sha256'] for r in old['images'])!=sorted(unique):raise ValueError('Refusing to mutate existing batch')
        else:save(path,doc)
        from .training import load_stream_manifest
        load_stream_manifest(path,'stream_sft',root=self.artifact_root)
        self.state['consumed']=sorted(consumed|{r['sha256'] for r in new_rows})
        self.state['batches'].append({'id':batch_id,'manifest':str(path),'images':len(selected),'new_images':len(new_rows),'created_at':utcnow()})
        for seed in self.settings['seeds']:
            self.state['jobs'].append({'id':batch_id+f'-{seed}','batch_id':batch_id,'manifest':str(path),'seed':seed,'status':'queued','attempts':0})
        self._save()

    def ingest_job_requests(self):
        """Admit only the bounded, frozen SFT source-control protocol."""
        from .ablations import KIND,ARMS
        from .training import load_stream_manifest
        for path in sorted((self.root/'job-requests').glob('*.json')):
            revision=file_hash(path)
            if revision in self.state['job_requests']:continue
            document=self._read(path,{})
            try:
                family=document.get('family_id','')
                if document.get('kind')!=KIND or not re.fullmatch('[a-f0-9]{24}',family):raise ValueError('Unsupported job request protocol')
                specs=document.get('jobs',[])
                if len(specs)!=6 or {(s.get('arm'),s.get('seed')) for s in specs}!={(a,s) for a in ARMS for s in (17,43,91)}:raise ValueError('Ablation requires exactly two arms and three seeds')
                pending=sum(j['status'] in ('queued','running') for j in self.state['jobs'])
                if pending+6>self.settings['maximum_queued_jobs']:continue
                admitted=[]
                for spec in specs:
                    relative=Path(spec['manifest']);manifest=(self.artifact_root/relative).resolve()
                    if relative.is_absolute() or not manifest.is_relative_to(self.artifact_root):raise ValueError('Ablation manifest escapes artifact root')
                    rows,protocol=load_stream_manifest(manifest,'stream_sft',root=self.artifact_root);metadata=self._read(manifest,{})['metadata']
                    if (metadata.get('experiment_branch')!=KIND or metadata.get('ablation_family_id')!=family or metadata.get('ablation_arm')!=spec['arm']
                        or metadata.get('batch_id')!=spec.get('batch_id') or spec.get('sft_steps')!=math.ceil(len(rows)/4)
                        or not 1<=spec['sft_steps']<=1000):raise ValueError('Frozen ablation identity, count or epoch differs')
                    admitted.append({'id':'ablation-'+spec['batch_id']+'-'+str(spec['seed']),'kind':KIND,'family_id':family,
                        'arm':spec['arm'],'batch_id':spec['batch_id'],'manifest':str(manifest),'seed':spec['seed'],'sft_steps':spec['sft_steps'],
                        'training_namespace':f"incremental-ablations-v1/{family}/{spec['arm']}/{spec['batch_id']}",
                        'status':'queued','attempts':0})
                existing={j['id'] for j in self.state['jobs']}
                self.state['jobs'].extend(j for j in admitted if j['id'] not in existing)
                self.state['job_requests'][revision]={'status':'admitted','family_id':family,'jobs':len(admitted),'path':str(path),'at':utcnow()}
            except (OSError,KeyError,ValueError,TypeError) as exc:self.state['job_requests'][revision]={'status':'rejected','reason':str(exc),'path':str(path),'at':utcnow()}
            self._save()

    def publish(self,error=None):
        rows,sessions,certs,source=self._catalog();running=[j for j in self.state['jobs'] if j['status']=='running'];training=[]
        for job in running:
            progress=self._read(Path(job['output'])/'status.json',{});stage=progress.get('stage','starting');slug='google_gemma-4-12B-it'
            if stage in ('baseline_evaluation','development_evaluation'):
                child='parent' if stage=='baseline_evaluation' else 'sft' if job.get('kind') else 'candidate'
                evaluation=self._read(Path(job['output'])/'evaluation'/child/'status.json',{})
                progress.update(evaluation_completed=evaluation.get('completed'),evaluation_total=evaluation.get('total'))
            namespace=job.get('training_namespace',f"incremental-stream-v1/{job['batch_id']}")
            metrics_path=self.artifact_root/'training'/namespace/slug/('stream_grpo' if not job.get('kind') and stage in ('grpo','development_evaluation','baseline_evaluation') else 'stream_sft')/f"seed-{job['seed']}"/'metrics.jsonl'
            metrics=[]
            if metrics_path.is_file():
                try:metrics=[json.loads(line) for line in metrics_path.read_text().splitlines()[-50:]]
                except ValueError:pass
            from .stream_training import reward_signal_diagnostics
            signal=reward_signal_diagnostics(metrics_path) if metrics_path.parent.parent.name=='stream_grpo' else None
            training.append({'job_id':job['id'],'seed':job['seed'],'gpu_group':job['gpu_group'],'stage':stage,'metrics':metrics,'reward_signal':signal,'kind':job.get('kind','stream_sft_grpo'),'arm':job.get('arm'),**progress})
        media=[]
        from PIL import Image,ImageDraw
        training_rows=[r for r in rows if r.get('split')=='train'];memberships={}
        for batch in self.state['batches']:
            for member in self._read(batch['manifest'],{'images':[]})['images']:memberships[member['sha256']]=batch['id']
        chosen=[r for r in training_rows if r.get('boxes') and r.get('provenance')=='direct_physics'][-4:]
        chosen+=[r for r in training_rows if r.get('boxes') and r.get('provenance')=='user_accepted_model'][-4:]
        chosen+=[r for r in training_rows if r.get('boxes') and r.get('provenance') in ('original_manual','reviewed_ai')][-4:]
        for row in chosen:
            relative=Path('pipeline/media')/(row['sha256']+'.jpg');path=self.artifact_root/relative
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(self.artifact_root/row['image']) as im:image=im.convert('RGB')
                draw=ImageDraw.Draw(image)
                for box in row.get('boxes',[]):
                    box=box.get('bbox',box);draw.rectangle((box['x_min']*image.width/1000,box['y_min']*image.height/1000,box['x_max']*image.width/1000,box['y_max']*image.height/1000),outline='#39dc7e',width=2)
                image.save(path,quality=90)
            media.append({'image':str(relative),'provenance':row['provenance'],'boxes':row.get('boxes',[]),'id':row['id'],
                'title':'Accepted physical patch' if row['provenance']=='direct_physics' else 'Reviewed/native training evidence',
                'research_accepted':row.get('research_accepted',False),'batch_id':memberships.get(row['sha256'])})
        for job in self.state['evaluation_jobs']:
            if job['status']=='running':
                progress=self._read(Path(job['output'])/'status.json',{})
                training.append({'job_id':job['id'],'seed':job['seed'],'gpu_group':job['gpu_group'],'stage':'sft_development_evaluation','metrics':[],
                    'evaluation_completed':progress.get('completed'),'evaluation_total':progress.get('total')})
        counts={status:sum(j['status']==status for j in self.state['jobs']) for status in ('queued','running','completed','failed')}
        status={'schema_version':1,'namespace':self.settings['namespace'],'updated_at':utcnow(),'status':'running' if training else 'waiting_for_accepted_arrivals',
            'stages':{'physics':{'status':'running' if not self.state['accepted_pairs'] else 'accepted_pairs_available','accepted_pairs':len(self.state['accepted_pairs']),'rejected_pairs':len(self.state['rejected_pairs'])},
                'synthetic':{'status':'completed' if self.state['accepted_pairs'] else 'blocked','images':sum(x['images'] for x in self.state['accepted_pairs'].values()),'detail':'Individual accepted patches; no unvalidated64m composition'},
                'native':{'status':'ready' if rows else 'waiting','eligible_images':len(rows),'input_counts':source.get('counts',{})},
                'training':{'status':'running' if training else 'waiting','active':training}},
            'queue':counts,'progress':{'batches':len(self.state['batches']),'consumed_new_images':len(self.state['consumed']),'seeds':self.settings['seeds']},
            'jobs':self.state['jobs'],'media':media,'promoted':self.state['promoted'],'deadline_utc':self.settings['deadline_utc'],
            'full_research_protocol':{'physical_tasks':184,'synthetic_images':30000,'seeds':[17,43,91],'status':'unchanged_separate_incomplete'},
            'dispatch_blocked':self.state.get('dispatch_blocked'),'last_error':error,
            'acceptance_note':'Experimental fields excluded; native machine labels are user-accepted, not human-reviewed'}
        status['stages']=[{'id':name,'label':name.replace('_',' ').title(),'status':value['status'],
            'completed':value.get('images',value.get('accepted_pairs',value.get('eligible_images',len(running)))),
            'total':None,'detail':value.get('detail',''),**value} for name,value in status['stages'].items()]
        status['queue_counts']=counts
        status['queue']=[{'id':j['id'],'kind':j.get('kind','stream_sft_grpo_evaluation'),'arm':j.get('arm'),'status':j['status'],'reason':j.get('reason'),'gpu_ids':j.get('gpu_group',[])} for j in self.state['jobs']]
        active_by_id={item['job_id']:item for item in training}
        for record,job in zip(status['queue'],self.state['jobs']):
            if not job.get('kind'):continue
            active=active_by_id.get(job['id'],{});last=(active.get('metrics') or [{}])[-1]
            record.update(seed=job['seed'],total=job['sft_steps'],completed=job['sft_steps'] if job['status']=='completed' else last.get('step',0),
                detail=f"{job['arm'].replace('_',' ')} · seed {job['seed']} · "+('GPU '+','.join(map(str,job['gpu_group'])) if job.get('gpu_group') else 'queued GPU lease'))
        status['queue']+=[{'id':j['id'],'kind':'sft_checkpoint_development_evaluation','status':j['status'],'reason':j.get('reason'),'gpu_ids':j.get('gpu_group',[])} for j in self.state['evaluation_jobs']]
        status['queue'].sort(key=lambda item:({'completed':0,'failed':1,'queued':2,'running':3}.get(item['status'],1),item['id']))
        status['stats']={'synthetic_images':sum(x['images'] for x in self.state['accepted_pairs'].values()),'accepted_pairs':len(self.state['accepted_pairs']),
            'training_batches':len(self.state['batches']),'training_completed':counts['completed'],'native_eligible':len(rows)}
        status['training']={'active':training,'metrics':[{'step':m.get('step'),'loss':m.get('loss'),
            'reward':m.get('reward',m.get('rewards/mean')),'seed':job['seed']} for job in training for m in job.get('metrics',[])]}
        status['gpu_probe']=self.state.get('gpu_probe',{'status':'planned_at_next_free_lease'})
        status['measured_topology']=self.state.get('measured_topology',{'gpu_groups':self.settings['gpu_groups'],'microbatch':1,'gradient_accumulation':4})
        status['arrival_gate']=self.state.get('arrival_gate',{})
        status['external_gpu_leases']=self.state.get('external_gpu_leases',[])
        save(self.root/'status.json',status);return status

    def tick(self):
        self.ingest_pairs();self.freeze_arrivals();self.ingest_job_requests();self.manage_jobs();return self.publish()

    def run_forever(self):
        with (self.root/'.worker.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while datetime.now(timezone.utc)<datetime.fromisoformat(self.settings['deadline_utc']).astimezone(timezone.utc):
                try:self.tick()
                except Exception as exc:
                    self.state['last_error']=str(exc)[:2000];self._save()
                    try:self.publish(str(exc)[:2000])
                    except Exception:save(self.root/'status.json',{'status':'failed_tick','updated_at':utcnow(),'error':str(exc)[:2000]})
                time.sleep(max(1,self.settings['poll_seconds']))

    def _group_free(self,group):
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            used={int(line.split(',')[0]):int(line.split(',')[1]) for line in result.stdout.splitlines()}
            return all(used.get(g,999999)<1024 for g in group)
        except (OSError,ValueError,subprocess.CalledProcessError):return False

    def _start(self,job,group):
        output=self.root/'runs'/job['id'];output.mkdir(parents=True,exist_ok=True)
        args=[sys.executable,'-u','-m','aerorozum.stream_training','--batch',job['manifest'],'--validation',str(self.root/'validation.json'),
            '--output',str(output),'--seed',str(job['seed']),'--sft-steps',str(self.settings['sft_steps']),'--grpo-steps',str(self.settings['grpo_steps'])]
        warm=self.state['promoted'].get(str(job['seed']))
        if warm:args+=['--warm-start',str(self.artifact_root/warm['adapter'])]
        if job.get('kind'):
            args=[sys.executable,'-u','-m','aerorozum.ablations','--batch',job['manifest'],'--validation',str(self.root/'validation.json'),
                '--output',str(output),'--seed',str(job['seed']),'--steps',str(job['sft_steps']),'--namespace',job['training_namespace']]
        env={**os.environ,'AEROROZUM_DATA_ROOT':str(self.artifact_root),'CUDA_VISIBLE_DEVICES':','.join(map(str,group)),
            'OMP_NUM_THREADS':str(self.settings['train_threads']),'MKL_NUM_THREADS':str(self.settings['train_threads']),
            'OPENBLAS_NUM_THREADS':str(self.settings['train_threads']),'TOKENIZERS_PARALLELISM':'false',
            'AEROROZUM_LOCAL_MODEL_PATH':self.settings.get('model_path','/opt/aerorozum/models/gemma-4-12B-it')}
        topology=self.state.get('measured_topology',{})
        env.update(AEROROZUM_MICROBATCH=str(topology.get('microbatch',1)),AEROROZUM_GRAD_ACCUMULATION=str(topology.get('gradient_accumulation',4)))
        if len(group)==1:env['AEROROZUM_GPU_MEMORY_FRACTION']='.88'
        if self.executor is not None:process=self.executor(args,env,output,job)
        else:
            stream=(output/'worker.log').open('ab');process=subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT);stream.close()
        self.processes[job['id']]=process
        job.update(status='running',pid=process.pid,gpu_group=group,output=str(output),started_at=utcnow(),attempts=job['attempts']+1)
        self._save()

    def manage_probe(self,controls,occupied):
        probe=self.state.get('gpu_probe',{})
        if probe.get('status')=='running':
            result=self._read(Path(probe['output'])/'result.json',{})
            try:os.kill(probe['pid'],0);alive=True
            except OSError:alive=False
            process=self.processes.get('gpu_probe')
            if process is not None:alive=process.poll() is None
            if result.get('status') in ('passed','failed') and not alive:
                selected=result.get('selected') or {}
                if result['status']=='passed' and result.get('single_gpu') is True and selected.get('memory_fraction',2)<=.9 and selected.get('forward_backward_verified') and selected.get('grpo_verified'):
                    self.state['measured_topology']={'gpu_groups':[[0],[1],[2],[3]],'microbatch':selected['microbatch'],
                        'gradient_accumulation':selected['gradient_accumulation'],'effective_batch':4,'probe_result':str(Path(probe['output'])/'result.json')}
                probe.update(status=result['status'],finished_at=utcnow(),result=result)
            elif not alive:probe.update(status='failed',reason='Memory probe process exited without measurement result')
            else:occupied.update(probe['reserved_group'])
        if controls.get('request_single_gpu_probe') and not probe and self.state['batches']:
            free=next((group for group in self.settings['gpu_groups'] if not any(g in occupied for g in group) and (self.executor is not None or self._group_free(group))),None)
            if free is not None:
                output=self.root/'gpu-probe';output.mkdir(parents=True,exist_ok=True)
                env={**os.environ,'AEROROZUM_DATA_ROOT':str(self.artifact_root),'CUDA_VISIBLE_DEVICES':str(free[0]),
                    'AEROROZUM_LOCAL_MODEL_PATH':self.settings.get('model_path','/opt/aerorozum/models/gemma-4-12B-it'),
                    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
                args=[sys.executable,'-u','-m','aerorozum.gpu_probe','--batch',self.state['batches'][0]['manifest'],'--output',str(output)]
                if self.executor is not None:process=self.executor(args,env,output,{'id':'gpu_probe'})
                else:
                    stream=(output/'worker.log').open('ab');process=subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT);stream.close()
                self.processes['gpu_probe']=process;probe={'status':'running','pid':process.pid,'reserved_group':free,'output':str(output),'started_at':utcnow()}
                occupied.update(free);self.state['gpu_probe']=probe

    def manage_jobs(self):
        controls=self._read(self.root/'controls.json',{})
        for job in self.state['jobs']:
            if job['status']!='running':continue
            result_path=Path(job['output'])/'result.json';process=self.processes.get(job['id'])
            if result_path.exists():
                result=self._read(result_path,{})
                if self._valid_result(job,result):
                    job.update(status='completed',finished_at=utcnow(),result=str(result_path))
                    if job.get('kind'):
                        job['checkpoint_utility']=result['checkpoint_utility'];continue
                    from .stream_training import reward_signal_diagnostics
                    signal=reward_signal_diagnostics((self.artifact_root/result['grpo']['adapter']).parent/'metrics.jsonl')
                    job['reward_signal']=signal
                    job['promotion_pending_reason']='Compare the saved SFT checkpoint on the same fixed development set before selection'
                    continue
            if process is not None:alive=process.poll() is None
            else:
                try:os.kill(job['pid'],0);alive=True
                except (OSError,KeyError):alive=False
            if not alive:
                progress=self._read(Path(job['output'])/'status.json',{})
                job.update(status='failed' if progress.get('status')=='failed' else 'queued',reason=progress.get('error','Worker interrupted; resume checkpoints'))
        self._manage_evaluation_results()
        deadline=datetime.fromisoformat(self.settings['deadline_utc']).astimezone(timezone.utc)
        if (deadline-datetime.now(timezone.utc)).total_seconds()<1800:
            self.state['dispatch_blocked']='Less than30minutes remain before VM cap';self._save();return
        occupied={g for j in self.state['jobs'] if j['status']=='running' for g in j.get('gpu_group',[])}
        occupied.update(g for j in self.state['evaluation_jobs'] if j['status']=='running' for g in j.get('gpu_group',[]))
        self.state['external_gpu_leases']=[]
        for lease in controls.get('external_gpu_leases',[]):
            try:
                expires=datetime.fromisoformat(lease['expires_at']).astimezone(timezone.utc);devices=lease['gpu_ids']
                if expires<=datetime.now(timezone.utc) or not isinstance(devices,list) or any(type(g) is not int or g not in (0,1,2,3) for g in devices):continue
                status='waiting_for_training_boundary' if any(g in occupied for g in devices) else 'reserved'
                occupied.update(devices);self.state['external_gpu_leases'].append({**lease,'status':status})
            except (ValueError,TypeError,KeyError):continue
        self.manage_probe(controls,occupied)
        if controls.get('pause_dispatch'):self._save();return
        topology=self.state.get('measured_topology',{})
        groups=topology.get('gpu_groups',self.settings['gpu_groups'])
        for group in groups:
            if any(g in occupied for g in group):continue
            evaluation=next((j for j in self.state['evaluation_jobs'] if j['status']=='queued'),None)
            if evaluation is not None:
                if self.executor is None and not self._group_free(group):continue
                self._start_evaluation(evaluation,group);occupied.update(group);continue
            def scope(j):return (j.get('family_id'),j.get('arm'),j['seed'])
            active_scopes={scope(j) for j in self.state['jobs'] if j['status']=='running'}
            active_scopes.update((None,None,j['seed']) for j in self.state['evaluation_jobs'] if j['status'] in ('queued','running'))
            candidates=[j for j in self.state['jobs'] if j['status']=='queued' and scope(j) not in active_scopes]
            if not candidates:break
            if self.executor is None and not self._group_free(group):continue
            self._start(candidates[0],group);occupied.update(group)
        self._save()

    def _start_evaluation(self,job,group):
        output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
        args=[sys.executable,'-u','-m','aerorozum.stream_training','--evaluate-only','--adapter',str(self.artifact_root/job['adapter']),
            '--validation',str(self.root/'validation.json'),'--output',str(output),'--seed',str(job['seed'])]
        env={**os.environ,'AEROROZUM_DATA_ROOT':str(self.artifact_root),'CUDA_VISIBLE_DEVICES':','.join(map(str,group)),
            'AEROROZUM_LOCAL_MODEL_PATH':self.settings.get('model_path','/opt/aerorozum/models/gemma-4-12B-it'),
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
        if len(group)==1:env['AEROROZUM_GPU_MEMORY_FRACTION']='.88'
        if self.executor is not None:process=self.executor(args,env,output,job)
        else:
            stream=(output/'worker.log').open('ab');process=subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT);stream.close()
        self.processes[job['id']]=process;job.update(status='running',pid=process.pid,gpu_group=group,started_at=utcnow());self._save()

    def _manage_evaluation_results(self):
        """Retroactively compare SFT for existing passes without touching workers."""
        known={j['parent_job_id'] for j in self.state['evaluation_jobs']}
        for job in self.state['jobs']:
            if job.get('kind') or job['status']!='completed' or job['id'] in known:continue
            result=self._read(job['result'],{})
            if not self._valid_result(job,result):continue
            self.state['evaluation_jobs'].append({'id':'sft-evaluation-'+job['id'],'parent_job_id':job['id'],'seed':job['seed'],
                'adapter':result['sft']['adapter'],'output':str(Path(job['output'])/'evaluation'/'sft-explicit-cache-v2'),'status':'queued'})
        for job in self.state['evaluation_jobs']:
            if job['status']!='running':continue
            summary=self._read(Path(job['output'])/'summary.json',{})
            process=self.processes.get(job['id'])
            if process is not None:alive=process.poll() is None
            else:
                try:os.kill(job['pid'],0);alive=True
                except OSError:alive=False
            expected=len(self._read(self.root/'validation.json',{'images':[]})['images'])
            if summary.get('status')=='completed' and summary.get('examples')==expected:
                from .stream_training import checkpoint_utility
                parent=next(j for j in self.state['jobs'] if j['id']==job['parent_job_id']);result=self._read(parent['result'],{})
                utility=checkpoint_utility(result['baseline'],summary,result['evaluation'],parent.get('reward_signal',{}))
                save(Path(parent['output'])/'checkpoint-utility.json',{'baseline':result['baseline'],'sft':summary,'grpo':result['evaluation'],'selection':utility})
                stage=utility['selected_stage']
                if stage:self.state['promoted'][str(parent['seed'])]={'adapter':result[stage]['adapter'],'batch_id':parent['batch_id'],
                    'f2':(summary if stage=='sft' else result['evaluation'])['micro_f2'],'stage':stage,'rl_improvement_claimed':False}
                parent['checkpoint_utility']=utility;parent.pop('promotion_pending_reason',None);job.update(status='completed',finished_at=utcnow())
            elif not alive:
                progress=self._read(Path(job['output'])/'status.json',{})
                job.update(status='failed' if progress.get('status')=='failed' else 'queued',reason=progress.get('error','Interrupted development evaluation; per-image cache resumes'))

    def _valid_result(self,job,result):
        if result.get('status')!='completed' or result.get('batch_id')!=job['batch_id'] or result.get('seed')!=job['seed']:return False
        batch=self._read(job['manifest'],{});expected_hash=digest(batch)
        stages=(('stream_sft','sft'),) if job.get('kind') else (('stream_sft','sft'),('stream_grpo','grpo'))
        for stage,key in stages:
            summary=result.get(key,{})
            if summary.get('status')!='completed' or summary.get('stage')!=stage or summary.get('manifest_hash')!=expected_hash or summary.get('seed')!=job['seed']:return False
            early=(key=='grpo' and summary.get('early_stopped') is True and summary.get('requested_optimizer_updates')==self.settings['grpo_steps']
                and summary.get('optimizer_updates',0)>=4 and isinstance(summary.get('early_stop_reason'),str) and 'zero-variance' in summary['early_stop_reason'])
            required=job.get('sft_steps',self.settings['sft_steps']) if key=='sft' else self.settings['grpo_steps']
            if (summary.get('optimizer_updates',0)<required and not early) or summary.get('nonzero_gradient_checks',0)<1:return False
            adapter=(self.artifact_root/summary.get('adapter','')).resolve()
            if not adapter.is_relative_to(self.artifact_root) or not (adapter/'adapter_config.json').is_file() or not (adapter/'adapter_model.safetensors').is_file():return False
        expected=len(self._read(self.root/'validation.json',{'images':[]})['images'])
        return all(result.get(k,{}).get('status')=='completed' and result[k].get('examples')==expected for k in ('baseline','evaluation'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--settings');p.add_argument('--once',action='store_true');a=p.parse_args()
    settings=json.loads(Path(a.settings).read_text()) if a.settings else None;worker=StreamCoordinator(a.root,settings)
    if a.once:print(json.dumps(worker.tick()))
    else:worker.run_forever()

if __name__=='__main__':main()
