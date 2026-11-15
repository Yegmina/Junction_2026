"""Incremental, hash-verified real inputs and explicitly machine-screened textures."""
from __future__ import annotations
from collections import Counter
import hashlib,json,math,sqlite3
from pathlib import Path
from PIL import Image,ImageStat
from .common import config,data_root,digest,save,utcnow
from .labelling import MODEL,classify_response
ZONE_SPLITS={**{f'zone{i}':'train' for i in range(1,6)},'zone7':'validation','zone8':'test','zone9':'test'}

class RealIngester:
 def __init__(self,root=None):
  self.root=Path(root or data_root()).resolve();self.output=self.root/'pipeline/inputs';self.output.mkdir(parents=True,exist_ok=True)
  self.db=sqlite3.connect(self.output/'ingest-cache.sqlite3');self.db.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, stamp TEXT, sha TEXT, stats TEXT)');self.verified={};self.documents={}
 def read(self,path):
  path=Path(path);stat=path.stat();stamp=(stat.st_ino,stat.st_ctime_ns,stat.st_mtime_ns,stat.st_size)
  if str(path) not in self.documents or self.documents[str(path)][0]!=stamp:self.documents[str(path)]=(stamp,json.loads(path.read_text()))
  return self.documents[str(path)][1]
 def image(self,value,sha,relative_to=None):
  path=Path(value);path=(path if path.is_absolute() else Path(relative_to or self.root)/path).resolve()
  if not path.is_relative_to(self.root) or not path.is_file():raise ValueError('Image missing or outside artifact root')
  stat=path.stat();stamp=f'{stat.st_ino}:{stat.st_ctime_ns}:{stat.st_mtime_ns}:{stat.st_size}';cached=self.verified.get(str(path))
  if cached is None:
   row=self.db.execute('SELECT stamp,sha,stats FROM files WHERE path=?',(str(path),)).fetchone()
   if row:cached=(row[0],row[1],json.loads(row[2]))
  if not cached or cached[0]!=stamp:
   actual=hashlib.sha256(path.read_bytes()).hexdigest()
   with Image.open(path) as im:
    gray=im.convert('L').resize((96,96));hist=gray.histogram();n=sum(hist);entropy=-sum((x/n)*math.log2(x/n) for x in hist if x)
    stats={'width':im.width,'height':im.height,'gray_stddev':ImageStat.Stat(gray).stddev[0],'entropy_bits':entropy,'radiometric':False}
   cached=(stamp,actual,stats);self.db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?)',(str(path),stamp,actual,json.dumps(stats)))
  self.verified[str(path)]=cached
  if cached[1]!=sha:raise ValueError('Image SHA256 mismatch')
  return path.relative_to(self.root).as_posix(),cached[2]
 def native(self,row,session_splits,labels_root=None):
  if row.get('status')=='invalid_model_response':raise ValueError('Invalid paid model response')
  if row.get('provenance')!='user_accepted_model' or row.get('model')!=MODEL or row.get('user_threshold')!=.6 or row.get('human_reviewed') is not False:raise ValueError('Exact user-authorized machine-label provenance required')
  source=row.get('source',{});sid=source.get('session_id')
  if not sid or session_splits.get(sid)!=row.get('split'):raise ValueError('Whole-session split mismatch')
  if not isinstance(source.get('video_sha256'),str) or len(source['video_sha256'])!=64:raise ValueError('Native video hash required')
  classification=classify_response(row.get('model_output'))
  if classification['uncertain'] or row.get('uncertain_boxes') or row.get('training_excluded_reason'):raise ValueError('Uncertain or mixed frame excluded')
  if row.get('detections')!=row['model_output']['detections']:raise ValueError('Cached detections differ from original model output')
  positive=bool(classification['accepted']);zero=classification['zero_candidates']
  if not positive and not(row.get('zero_candidates') is True and zero):raise ValueError('No explicit zero-candidate response')
  boxes=row.get('boxes',[])
  if len(boxes)!=len(classification['accepted']) or any(b['bbox']!=d['bbox'] for b,d in zip(boxes,classification['accepted'])):raise ValueError('Accepted boxes mismatch')
  image,stats=self.image(row['image'],row['sha256'])
  result={k:row[k] for k in ('id','sha256','split','provenance','model','user_threshold','human_reviewed','source','timestamp_s','source_offset_s','frame_index','time_basis','receipt_timestamp_ns','wallclock_alignment_available','gps_alignment_available','timing_resolution','annotated','cloud_sync','labelled_at','label_prompt_version','label_prompt_sha256','prior_paid_inference') if k in row}
  result.update(image=image,boxes=boxes,zero_candidates=zero,detections=row['detections'],uncertain_boxes=[],row_kind='native_label',source_group='video:'+sid,image_statistics=stats,independently_verified_identity=False)
  if labels_root is not None and row.get('cache_key'):
   receipt_path=Path(labels_root)/'upload-receipts'/(row['cache_key']+'.json')
   if receipt_path.exists():
    from .label_sync import frame_identity
    receipt=self.read(receipt_path)
    if receipt.get('status')=='uploaded' and receipt.get('frame_identity')==frame_identity(row):result['cloud_sync']={'status':'uploaded','uris':receipt['uris'],'verification_receipt':receipt_path.relative_to(self.root).as_posix(),'source_record_unchanged':True}

  return result
 def scan(self):
  native_roots=[p for p in (self.root/'labelling').glob('*/labels') if p.parent.name in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne')];session_splits={};quarantine=[];images=[];excluded=[];failures=[];completed_checkpoints=set();cache_counts=Counter();paid_identities=set()
  flight_path=self.root/'pipeline/inputs/airborne-eligibility.json';flight=self.read(flight_path) if flight_path.exists() else None
  flight_required=flight is not None or (self.root/'labelling/airborne-only-drain.json').exists()
  flight_decisions={(f.get('source',{}).get('object'),str(f.get('source',{}).get('generation')),f.get('frame_index')):f['eligibility'] for f in (flight or {}).get('frames',[])}
  for labels in native_roots:
   registry=labels.parent/'session-splits.json'
   if registry.exists():
    mapping=self.read(registry).get('sessions',{})
    for sid,split in mapping.items():
     if sid in session_splits and session_splits[sid]!=split:raise ValueError('Conflicting frozen session registries')
     session_splits[sid]=split
   sampling=labels.parent/'sampling-manifest.json'
   if sampling.exists():quarantine.extend(self.read(sampling).get('failed_sources',[]))
  for labels in native_roots:
   for directory,name in [('results','result_records'),('paid-responses','paid_response_records'),('provider/responses','provider_response_records'),('provider/errors','provider_error_records')]:cache_counts[name]+=sum(1 for _ in (labels/directory).glob('*.json'))
   for path in (labels/'results').glob('*.json'):
    try:
     row=self.read(path)
     if row.get('model')==MODEL and row.get('sha256'):paid_identities.add((row['sha256'],row.get('source',{}).get('generation')))
     result=self.native(row,session_splits,labels);source=result['source'];decision=flight_decisions.get((source.get('object'),str(source.get('generation')),result.get('frame_index')))
     result['flight_state']='airborne' if decision and decision['status']=='eligible' else 'known_ground' if decision and decision['status']=='ground' else 'unknown' if decision else 'unclassified'
     result['flight_eligibility']=decision or {'status':'unclassified','reason':'not_yet_classified_by_telemetry_filter','airborne_eligible':False}
     images.append(result)
    except (ValueError,KeyError,TypeError,OSError) as ex:excluded.append({'id':locals().get('row',{}).get('id'),'cache':path.relative_to(self.root).as_posix(),'reason':str(ex)[:200]})
   for path in (labels/'checkpoints').glob('*.json'):
    entry=self.read(path)
    if entry.get('status')=='completed':completed_checkpoints.add(entry.get('id'))
    if entry.get('status')=='failed':failures.append({**{k:entry.get(k) for k in ('id','sha256','error','category','phase','retryable','observed_at','at','paid_request_not_issued')},'source':entry.get('source') or entry.get('row',{}).get('source')})
  real_manifest=self.root/'datasets/real/manifest.json';manual=self.read(real_manifest) if real_manifest.exists() else {'images':[]};parents={Path(r.get('archive_image','')).name:r for r in manual['images']}
  crop_manifest=self.root/'backgrounds/annotation-excluded-train/manifest.json';crop_doc=self.read(crop_manifest) if crop_manifest.exists() else {};archive_sha=crop_doc.get('source_archive_sha256')
  for row in manual['images']:
   try:
    if row.get('provenance') not in ('original_manual','reviewed_ai') or ZONE_SPLITS.get(row.get('zone'))!=row.get('split') or not row.get('boxes'):raise ValueError('Invalid approved manual anchor')
    image,stats=self.image(row['image'],row['sha256']);source={'source_archive_sha256':archive_sha,'parent_image_sha256':row['sha256'],'archive_member':row['archive_image'],'zone':row['zone']}
    images.append({**row,'image':image,'source':source,'source_group':f'archive:{archive_sha}:{row["zone"]}','row_kind':'manual_anchor','zero_candidates':False,'human_reviewed':row['provenance']=='original_manual','image_statistics':stats,'independently_verified_identity':False})
   except (ValueError,KeyError,OSError) as ex:excluded.append({'id':row.get('id'),'reason':str(ex)[:200]})
  for crop in crop_doc.get('crops',[]):
   screen=crop.get('screening',{})
   if not(crop.get('split')=='train' and ZONE_SPLITS.get(crop.get('parent_zone'))=='train' and screen.get('model')==MODEL and screen.get('status')=='accepted' and screen.get('decision')=='no_mine_detected' and screen.get('human_reviewed') is False and screen.get('rule')=='zero_candidates_at_any_confidence' and screen.get('detections')==[] and crop.get('detections')==[]):continue
   try:
    image,stats=self.image(crop['image'],crop['sha256'],crop_manifest.parent);parent=parents[crop['parent_image']]
    if parent['sha256']!=crop['parent_sha256']:raise ValueError('Crop parent hash mismatch')
    images.append({'id':crop['id'],'image':image,'sha256':crop['sha256'],'split':'train','provenance':'user_accepted_model','boxes':[],'detections':[],'uncertain_boxes':[],'zero_candidates':True,'model':MODEL,'user_threshold':.6,'human_reviewed':False,'source_group':f'archive:{archive_sha}:{crop["parent_zone"]}',
     'source':{'source_archive_sha256':archive_sha,'parent_image_sha256':crop['parent_sha256'],'archive_member':parent['archive_image'],'zone':crop['parent_zone'],'crop_xyxy':crop['crop_xyxy'],'excluded_boxes_with_margin_px':crop['excluded_boxes_with_margin_px'],'minimum_margin_px':crop['minimum_margin_px']},'row_kind':'zip_screened_background','screening':screen,'image_statistics':stats,'independently_verified_identity':False})
   except (ValueError,KeyError,OSError) as ex:excluded.append({'id':crop.get('id'),'reason':str(ex)[:200]})
  split_hashes={}
  for row in images:split_hashes.setdefault(row['sha256'],set()).add(row['split'])
  for row in images:
   row['cross_split_duplicate']=len(split_hashes[row['sha256']])>1
   row['training_eligible']=row['split']=='train' and not row['cross_split_duplicate'] and not(flight_required and row['row_kind']=='native_label' and row['flight_state']!='airborne')
   if flight_required and row['row_kind']=='native_label' and row['flight_state']!='airborne':row['training_excluded_reason']='airborne_only_policy_'+row['flight_state']
  unique={}
  for row in images:unique.setdefault((row['id'],row['sha256'],row['split']),row)
  images=sorted(unique.values(),key=lambda x:(x['source_group'],x['id']));background_groups={}
  for row in images:
   stats=row['image_statistics']
   if not(row['training_eligible'] and row.get('zero_candidates') is True and row.get('detections')==[]):continue
   if row['row_kind']=='native_label' and not(stats['gray_stddev']>=3 and stats['entropy_bits']>=2 and stats['width']>=128 and stats['height']>=96):continue
   screen={'status':'accepted','decision':'no_mine_detected','model':MODEL,'detections':[],'rule':'zero_candidates_at_any_confidence','human_reviewed':False,'verified_negative':False}
   background={**row,'screening':screen,'target_free_verified':False,'background_identity_status':'machine_screened_no_candidate_not_verified_negative','background_source':'annotation_excluded_crop' if row['row_kind']=='zip_screened_background' else 'bucket_frame_explicit_zero_candidates'}
   background_groups.setdefault(row['source_group'],[]).append(background)
  backgrounds=[]
  while any(background_groups.values()):
   for group in sorted(background_groups):
    if background_groups[group]:backgrounds.append(background_groups[group].pop())
  counts={'images':len(images),'native_labels':sum(r['row_kind']=='native_label' for r in images),'manual_anchor_train':sum(r['row_kind']=='manual_anchor' and r['training_eligible'] for r in images),'zip_screened_backgrounds':sum(r['row_kind']=='zip_screened_background' for r in images),
   'positive_frames':sum(bool(r['boxes']) for r in images),'explicit_zero_frames':sum(r.get('zero_candidates') is True for r in images),'train_positive_frames':sum(r['training_eligible'] and bool(r['boxes']) for r in images),'train_zero_frames':sum(r['training_eligible'] and r.get('zero_candidates') is True for r in images),
   'native_train_positive_frames':sum(r['row_kind']=='native_label' and r['training_eligible'] and bool(r['boxes']) for r in images),'native_train_zero_frames':sum(r['row_kind']=='native_label' and r['training_eligible'] and r.get('zero_candidates') is True for r in images),
   'background_bank':len(backgrounds),'excluded_records':len(excluded),'unresolved_frame_failures':len(failures),'quarantined_sources':len(quarantine),'cross_split_duplicate_rows':sum(r['cross_split_duplicate'] for r in images),'splits':dict(Counter(r['split'] for r in images))}
  counts.update({'native_'+state:sum(r['row_kind']=='native_label' and r.get('flight_state')==state for r in images) for state in ('airborne','known_ground','unknown','unclassified')})
  cloud=Counter(r.get('cloud_sync',{}).get('status','not_requested') for r in images if r['row_kind']=='native_label')
  recovery_path=self.root/'labelling/robotics-er2-5fps/recovery/status.json';recovery=dict(self.read(recovery_path)) if recovery_path.exists() else None
  if recovery is not None:
   frozen_path=recovery_path.parent/'manifest.json'
   if frozen_path.exists():
    recovered_ids={r['id'] for r in self.read(frozen_path)['images']}
    recovery.update(completed=len(recovered_ids & completed_checkpoints),unresolved=len(recovered_ids-completed_checkpoints),progress_observed_at=utcnow())
  document={'schema_version':1,'updated_at':utcnow(),'images':images,'counts':counts,'session_splits':session_splits,'zone_splits':ZONE_SPLITS,
   'cache_counts':dict(cache_counts),'cloud_sync':dict(cloud),'failure_reasons':dict(Counter(f.get('category','other') for f in failures)),
   'frame_failures':'pipeline/inputs/frame-failures.json','excluded_records':'pipeline/inputs/excluded.json','background_manifest':'pipeline/inputs/backgrounds.json','quarantined_sources':quarantine,'recovery':recovery,
   'source_assets':[r for r in images if r['row_kind']=='native_label' and r['boxes']][-8:]+[r for r in images if r['row_kind']=='manual_anchor' and r['training_eligible']][:4],
   'limitations':['Machine zero-candidate responses are not independently verified absence.','Unlabelled flight images are excluded from this training/background index.','Full-image ZIP positives never become negative backgrounds.','Held-out rows retain their original split and are never eligible training textures.']}
  document['airborne_only_required']=flight_required
  if flight:
   pending=[f for f in flight['frames'] if f['eligibility']['status']=='eligible' and (f['sha256'],f.get('source',{}).get('generation')) not in paid_identities]
   document['flight_filter']={'policy':flight['policy'],'counts':{**flight['counts'],'airborne_pending_frames':len(pending),'airborne_prior_paid_frames':flight['counts']['airborne_eligible']-len(pending),'estimated_new_unique_inference_requests':len({(f['sha256'],f['source']['generation']) for f in pending})},'audit':'pipeline/inputs/airborne-eligibility.json','eligible_manifest':'pipeline/inputs/airborne-eligible.json','paid_inference_issued':False}
  for key,path in [('credential_cache_maintenance',self.root/'labelling/sdk-token-cache-status.json'),('main_failure_observer',self.root/'labelling/robotics-er2-5fps/recovery/main-failure-observer.json')]:
   if path.exists():document[key]=self.read(path)
  timing_path=self.root/'labelling/quarantine-recovery/timing-resolution.json'
  if timing_path.exists():
   document['source_timing_recovery']=self.read(timing_path)
   document['counts']['recovered_receipt_segments']=document['source_timing_recovery']['metadata']['receipt_aligned_segments']
   document['counts']['video_time_only_segments']=document['source_timing_recovery']['metadata']['video_time_only_segments']
  queue_path=self.root/'labelling/quarantine-recovery/status.json'
  if queue_path.exists():document['quarantine_label_queue']=self.read(queue_path)
  additional=self.root/'labelling/additional-bucket-sources'
  if (additional/'watch.json').exists():
   coverage=self.read(additional/'watch.json')
   document['source_coverage']={k:v for k,v in coverage.items() if k!='pending'}
   counts.update(raw_thermal_videos=coverage['normal_raw_objects'],additional_source_versions_pending=coverage['pending_source_versions'])
  if (additional/'manifest.json').exists():
   prepared=self.read(additional/'manifest.json')
   document['additional_label_queue']={**prepared['metadata'],'status':'superseded_by_airborne_only_filter' if flight_required else prepared['metadata']['status'],'queue_order':'airborne eligible manifest only' if flight_required else 'after main and eight-source quarantine recovery','frozen_session_splits':True,'main_calls_interrupted':False}
   counts['additional_prepared_frames']=len(prepared['images'])
   counts['additional_receipt_aligned_segments']=sum(s.get('timing',{}).get('wallclock_alignment_available') is True for s in prepared['sources'])
  if (additional/'status.json').exists():document['additional_label_stage']=self.read(additional/'status.json')
  document['dataset_signature']=digest({'images':[(r['id'],r['sha256'],r['split'],r['boxes'],r['training_eligible']) for r in images],'session_splits':session_splits})
  bank={'schema_version':2,'image_root':'artifact_root','status':'screened','updated_at':document['updated_at'],'screening_model':MODEL,'accepted_crops':len(backgrounds),'crops':backgrounds,'target_free_verified':False,'allowed_splits':['train'],'limitation':'Only explicit zero-candidate machine screening; not human-confirmed mine-free ground.'}
  save(self.output/'backgrounds.json',bank);save(self.output/'frame-failures.json',{'updated_at':document['updated_at'],'count':len(failures),'failures':failures});save(self.output/'excluded.json',{'updated_at':document['updated_at'],'count':len(excluded),'records':excluded});save(self.output/'real.json',document);self.db.commit();return document
