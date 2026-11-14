"""Frozen paired Robotics prompts, selected exclusively on zone7 development."""
from __future__ import annotations
import argparse,asyncio,copy,json,math,time
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
from .common import ROOT,config,data_root,digest,save,utcnow,redacted_error
from .training import checked_box,detection_score
from .labelling import MODEL
from .label_prompts import PROMPTS
from .fast_fields import file_hash

VERSIONS=('thermal-v1-conservative','thermal-v2-recall-calibrated')
THRESHOLDS=(0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95)

def freeze_protocol(root,output,expected_images=88):
 root=Path(root).resolve();path=Path(output)/'protocol.json'
 if path.exists():
  protocol=json.loads(path.read_text())
  if protocol.get('protocol_sha256')!=digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}):raise ValueError('Frozen protocol changed')
  return protocol
 manifest=root/'datasets/real/manifest.json';document=json.loads(manifest.read_text())
 rows=[copy.deepcopy(r) for r in document['images'] if r.get('zone')=='zone7' and r.get('split')=='validation'];rows.sort(key=lambda r:r['id'])
 if len(rows)!=expected_images:raise ValueError(f'Expected {expected_images} zone7 development images, found {len(rows)}')
 for row in rows:
  image=(root/row['image']).resolve()
  if not image.is_relative_to(root) or file_hash(image)!=row['sha256']:raise ValueError('Development image path/hash mismatch')
  if not row.get('boxes'):raise ValueError('Unexpected non-positive scene; revise the protocol explicitly')
  for box in row['boxes']:checked_box(box.get('bbox',box))
 protocol={'schema_version':1,'model':MODEL,'versions':list(VERSIONS),'prompts':{v:PROMPTS[v] for v in VERSIONS},
  'generation':{'temperature':.1,'max_output_tokens':2048,'api_version':'v1beta','thinking_budget':0},
  'limits':{'requests_per_minute':120,'input_tokens_per_minute':500000,'concurrency':16,'max_requests':2*len(rows),'max_retries':1},
  'dataset_manifest_sha256':file_hash(manifest),'images':rows,'expected_images':len(rows),'thresholds':list(THRESHOLDS),'primary_confidence':.6,'iou_threshold':.5,
  'selection_scope':'zone7 development only; zones8/9 remain sealed','ground_truth_in_model_prompt':False,
  'annotation_provenance':dict(Counter(r['provenance'] for r in rows)),'background_fpr_estimable':False,'training_eligible':False,'created_at':utcnow()}
 protocol['protocol_sha256']=digest(protocol);save(path,protocol);return protocol

def candidates(output):
 if not isinstance(output,dict) or not isinstance(output.get('detections'),list):return [],1
 valid=[];invalid=0
 for item in output['detections']:
  try:
   confidence=item['confidence']
   if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
   valid.append({**item,'bbox':checked_box(item['bbox']),'confidence':float(confidence)})
  except (ValueError,TypeError,KeyError,OverflowError):invalid+=1
 return valid,invalid

def aggregate(protocol,records):
 summaries={};paired=[];missing=[]
 for row in protocol['images']:
  pair={version:records.get((row['id'],version)) for version in protocol['versions']}
  if any(not value or value.get('status')=='transport_error' for value in pair.values()):missing.append(row['id']);continue
  paired.append((row,pair))
 for version in protocol['versions']:
  curve=[]
  for threshold in protocol['thresholds']:
   scores=[];invalid=0;proposals=0
   for row,pair in paired:
    valid,bad=candidates(pair[version].get('model_output'));invalid+=bad;selected=[item['bbox'] for item in valid if item['confidence']>=threshold];proposals+=len(selected)
    scores.append(detection_score(selected,row['boxes'],protocol['iou_threshold']))
   tp=sum(s['tp'] for s in scores);fp=sum(s['fp'] for s in scores);fn=sum(s['fn'] for s in scores)
   curve.append({'confidence_threshold':threshold,'images':len(paired),'tp':tp,'unmatched_boxes':fp,'fn':fn,
    'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,
    'unmatched_boxes_per_image':fp/len(paired) if paired else None,'proposals':proposals,'invalid_candidate_records':invalid})
  summaries[version]={'at_0_6':next(x for x in curve if abs(x['confidence_threshold']-.6)<1e-8),'threshold_curve':curve,
   'valid_output_images':sum(candidates(pair[version].get('model_output'))[1]==0 for _,pair in paired)}
 winner=None
 if not missing and paired:
  ranked=sorted(summaries,key=lambda v:summaries[v]['at_0_6']['f2'],reverse=True)
  if summaries[ranked[0]]['at_0_6']['f2']>summaries[ranked[1]]['at_0_6']['f2']+1e-12:winner=ranked[0]
 return {'status':'completed' if not missing else 'incomplete','protocol_sha256':protocol['protocol_sha256'],'paired_images':len(paired),'missing_images':missing,
  'primary_confidence':.6,'iou_threshold':.5,'versions':summaries,'measured_winner_at_0_6':winner,'background_false_positive_rate':None,
  'interpretation':'All development frames are positive. Unmatched boxes measure disagreement with reviewed annotations, not a true background false-positive rate. One-zone development selection does not establish independent test performance.'}

async def run(output=None,provider=None,store=None,root=None):
 from .providers import GeminiProvider
 from .cloud_cache import GCSRecordStore
 started=time.monotonic();root=Path(root or data_root()).resolve();output=Path(output or root/'prompt-comparisons/zone7-v1-v2-er2').resolve();output.mkdir(parents=True,exist_ok=True)
 protocol=freeze_protocol(root,output);identity=protocol['protocol_sha256'];records={};failures=[];uploaded=0
 bucket=json.loads((ROOT/'configs/labelling.json').read_text())['gcs_bucket'];store=store or GCSRecordStore(bucket,prefix='robotics-er2-labels/prompt-evaluations/'+identity[:24])
 owned=provider is None;cfg={**config()['gemini'],**protocol['generation'],**protocol['limits'],'bulk_model':MODEL,'reserve_output_tokens':2048,'drain_path':str(output/'DRAIN')}
 provider=provider or GeminiProvider(cfg,output_dir=output/'provider')
 lease=root/'pipeline/leases/prompt-comparison.json'
 save(lease,{'status':'running','owner':identity,'model':MODEL,**protocol['limits'],'started_at':utcnow(),'expires_at':(datetime.now(timezone.utc)+timedelta(minutes=15)).isoformat()})
 save(output/'progress.json',{'status':'starting','completed':0,'total':2*len(protocol['images']),'at':utcnow()})
 await asyncio.to_thread(store.put,digest(['protocol',identity]),protocol,'progress')
 queue=asyncio.Queue()
 for index,row in enumerate(protocol['images']):
  for version in (VERSIONS if index%2==0 else VERSIONS[::-1]):queue.put_nowait((row,version))
 async def worker():
  nonlocal uploaded
  while not queue.empty():
   try:row,version=queue.get_nowait()
   except asyncio.QueueEmpty:return
   key=digest([identity,row['id'],row['sha256'],version]);path=output/'results'/(key+'.json');record=None
   try:
    if path.is_file():record=json.loads(path.read_text())
    else:record=await asyncio.to_thread(store.get,key)
    if record is not None and (record.get('protocol_sha256')!=identity or record.get('image_sha256')!=row['sha256'] or record.get('prompt_version')!=version):raise ValueError('Evaluation cache identity mismatch')
    if record is None:
     error=None;model_output=None
     try:model_output=await provider.generate('robotics_er2_prompt_comparison',protocol['prompts'][version],[root/row['image']],model=MODEL,max_output_tokens=2048)
     except Exception as ex:error=redacted_error(ex)
     raw=provider.response_record('robotics_er2_prompt_comparison',protocol['prompts'][version],[root/row['image']],model=MODEL,max_output_tokens=2048)
     valid,bad=candidates(model_output)
     record={'id':row['id'],'image':row['image'],'image_sha256':row['sha256'],'prompt_version':version,'protocol_sha256':identity,'model':MODEL,
      'model_output':model_output,'raw_response':raw,'detections':valid,'accepted_boxes':[d for d in valid if d['confidence']>=.6],
      'uncertain_boxes':[d for d in valid if d['confidence']<.6],'invalid_candidate_records':bad,
      'status':'completed' if model_output is not None else 'invalid_paid_response' if raw and raw.get('raw_text') is not None else 'transport_error',
      'error':error,'split':'validation','zone':'zone7','provenance':'development_prompt_evaluation','training_eligible':False,'at':utcnow()}
     save(path,record)
    records[(row['id'],version)]=record
    if record.get('cloud_sync',{}).get('status')!='uploaded':
     uris=await asyncio.to_thread(store.write_frame,key,record,record.get('raw_response'));record['cloud_sync']={'status':'uploaded','uris':uris};save(path,record)
    uploaded+=1
   except Exception as ex:failures.append({'id':row['id'],'version':version,'error':redacted_error(ex)});save(output/'failures.json',failures)
   finally:
    save(output/'progress.json',{'status':'running','completed':len(records),'total':2*len(protocol['images']),'cloud_synced':uploaded,'errors':len(failures),'issued':getattr(provider,'issued',None),'at':utcnow()});queue.task_done()
 try:await asyncio.gather(*(worker() for _ in range(16)))
 finally:
  if owned:await provider.close()
 report=aggregate(protocol,records);report.update(archive_failures=failures,cloud_synced=uploaded,actual_provider_attempts=getattr(provider,'issued',None),elapsed_s=time.monotonic()-started)
 if hasattr(provider,'limiter'):report['rate_limiter_snapshot']=await provider.limiter.snapshot()
 pairs=[]
 for row in protocol['images']:
  if all((row['id'],v) in records for v in VERSIONS):
   pair={v:records[(row['id'],v)] for v in VERSIONS};paired={'id':row['id'],'image_sha256':row['sha256'],'reviewed_boxes':row['boxes'],'predictions':{v:pair[v]['model_output'] for v in VERSIONS},'training_eligible':False}
   pairs.append(paired);save(output/'paired'/(row['id']+'.json'),paired)
 save(output/'paired-predictions.json',{'protocol_sha256':identity,'images':pairs,'training_eligible':False});save(output/'report.json',report)
 await asyncio.to_thread(store.put,digest(['paired',identity]),{'protocol_sha256':identity,'images':pairs,'training_eligible':False},'progress')
 await asyncio.to_thread(store.put,digest(['report',identity]),report,'progress')
 write_report(root,output,protocol,records,report)
 handoff=(datetime.now(timezone.utc)+timedelta(seconds=60)).isoformat();save(lease,{'status':'completed','owner':identity,'report':str(output/'report.json'),'resume_paid_requests_not_before':handoff,'at':utcnow()})
 save(output/'progress.json',{'status':report['status'],'completed':len(records),'total':2*len(protocol['images']),'cloud_synced':uploaded,'errors':len(failures),'at':utcnow()});return report

def annotate_pair(root,output,row,pair):
 from PIL import Image,ImageDraw
 with Image.open(Path(root)/row['image']) as source:image_source=source.convert('RGB')
 panels=[]
 for version in VERSIONS:
  image=image_source.copy();draw=ImageDraw.Draw(image)
  for target in row['boxes']:
   b=target.get('bbox',target);draw.rectangle((b['x_min']*image.width/1000,b['y_min']*image.height/1000,b['x_max']*image.width/1000,b['y_max']*image.height/1000),outline='#76ddff',width=2)
  for item in candidates(pair[version].get('model_output'))[0]:
   b=item['bbox'];color='#8de99e' if item['confidence']>=.6 else '#f2be72';draw.rectangle((b['x_min']*image.width/1000,b['y_min']*image.height/1000,b['x_max']*image.width/1000,b['y_max']*image.height/1000),outline=color,width=2)
   draw.text((b['x_min']*image.width/1000,max(0,b['y_min']*image.height/1000-12)),f"{item['confidence']:.2f}",fill=color)
  panel=Image.new('RGB',(image.width,image.height+28),'#142027');panel.paste(image,(0,28));ImageDraw.Draw(panel).text((8,8),version,fill='white');panels.append(panel)
 combined=Image.new('RGB',(image_source.width*2+8,image_source.height+28),'#31434a');combined.paste(panels[0],(0,0));combined.paste(panels[1],(image_source.width+8,0))
 path=Path(output)/'qualitative'/(row['id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True);combined.save(path,quality=94);return str(path)

def write_report(root,output,protocol,records,report):
 import csv
 differences=[]
 for row in protocol['images']:
  if all((row['id'],v) in records for v in VERSIONS):
   pair={v:records[(row['id'],v)] for v in VERSIONS};scores=[detection_score([b['bbox'] for b in candidates(pair[v]['model_output'])[0] if b['confidence']>=.6],row['boxes'])['f2'] for v in VERSIONS]
   low=sum(b['confidence']<.6 for v in VERSIONS for b in candidates(pair[v]['model_output'])[0]);differences.append((scores[1]-scores[0],low,row,pair))
 chosen=[];seen=set()
 groups=[sorted(differences,key=lambda x:x[0],reverse=True)[:4],sorted(differences,key=lambda x:x[0])[:4],sorted(differences,key=lambda x:x[1],reverse=True)[:4]]
 for group in groups:
  for delta,low,row,pair in group:
   if row['id'] in seen:continue
   seen.add(row['id']);chosen.append({'id':row['id'],'v2_minus_v1_f2':delta,'lower_confidence_proposals':low,'image':annotate_pair(root,output,row,pair)})
 save(output/'qualitative-selection.json',{'rule':'Largest improvements, regressions and lower-confidence proposal counts; illustrative, not representative sampling','examples':chosen})
 with (output/'threshold-curves.csv').open('w',newline='') as f:
  first=report['versions'][VERSIONS[0]]['threshold_curve'][0];writer=csv.DictWriter(f,fieldnames=['prompt_version',*first]);writer.writeheader()
  for version in VERSIONS:
   for row in report['versions'][version]['threshold_curve']:writer.writerow({'prompt_version':version,**row})
 lines=['# Frozen thermal prompt comparison','',f"Status: {report['status']}. Paired development images: {report['paired_images']}. Exact model: `{MODEL}`.",'', '| Prompt | Precision @0.6 | Recall @0.6 | F2 @0.6 | Unmatched boxes/image |','|---|---:|---:|---:|---:|']
 for version in VERSIONS:
  score=report['versions'][version]['at_0_6'];lines.append(f"| {version} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f2']:.4f} | {score['unmatched_boxes_per_image'] or 0:.4f} |")
 lines+=['',f"Measured development winner at the authorized 0.6 threshold: {report['measured_winner_at_0_6'] or 'no complete-set winner / tie'}.",'',report['interpretation'],'','Threshold curves, raw outputs, immutable prompt text and paired overlays are retained alongside this report. Cyan boxes are reviewed annotations, green are proposals >=0.6, amber are lower-confidence proposals.','',f"Reproduce: `python -m aerorozum.prompt_comparison --output '{output}'` (cache-first; frozen prompts retained)."]
 (output/'report.md').write_text('\n'.join(lines)+'\n')

def schema_diagnostic(protocol,records,normalize_aliases=False):
 """Strict production semantics on copies; original provider outputs stay intact."""
 from .labelling import classify_response
 from .label_schema import normalize_thermal_output
 derived={};rejected={v:[] for v in VERSIONS};traces=[]
 for key,record in records.items():
  output=copy.deepcopy(record.get('model_output'))
  try:
   if normalize_aliases:
    output,changes=normalize_thermal_output(output)
    if changes:traces.append({'id':key[0],'version':key[1],'changes':changes})
   classify_response(output,.6)
  except (ValueError,TypeError,KeyError):rejected[key[1]].append(key[0]);output={'detections':[]}
  derived[key]={**record,'model_output':output}
 result=aggregate(protocol,derived);result['whole_image_rejected_frames']=rejected;result['normalization_trace']=traces
 result['policy']='lossless_aliases_then_strict_whole_image' if normalize_aliases else 'strict_raw_whole_image'
 result['classifier_sha256']=file_hash(ROOT/'aerorozum/labelling.py');result['normalizer_sha256']=file_hash(ROOT/'aerorozum/label_schema.py')
 for version in VERSIONS:result['versions'][version]['valid_output_images']=len(protocol['images'])-len(rejected[version])
 return result

def finalize_report(output,archive=True):
 """Reanalyze saved responses and plot curves without invoking a model."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 output=Path(output);report=json.loads((output/'report.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
 if report['status']!='completed':raise ValueError('The fixed paired evaluation is not complete')
 records=sorted([json.loads(p.read_text()) for p in (output/'results').glob('*.json')],key=lambda r:(r['id'],r['prompt_version']))
 indexed={(r['id'],r['prompt_version']):r for r in records}
 report['strict_production_raw']=schema_diagnostic(protocol,indexed)
 report['lossless_aliases_then_strict']=schema_diagnostic(protocol,indexed,normalize_aliases=True)
 for version in VERSIONS:
  item=report['versions'][version];low=item['threshold_curve'][0];primary=item['at_0_6']
  item['lower_confidence_analysis']={'additional_matched_boxes_below_0_6':low['tp']-primary['tp'],
   'additional_unmatched_boxes_below_0_6':low['unmatched_boxes']-primary['unmatched_boxes'],
   'additional_proposals_below_0_6':low['proposals']-primary['proposals'],'acceptance_threshold_unchanged':.6}
  usages=[r.get('raw_response',{}).get('usage',{}).get('prompt_token_count') for r in records if r['prompt_version']==version]
  usages=[n for n in usages if type(n) is int]
  item['measured_prompt_tokens']={'samples':len(usages),'mean':sum(usages)/len(usages) if usages else None,'total':sum(usages)}
 figure,axes=plt.subplots(1,2,figsize=(11,4.2))
 for version,color in zip(VERSIONS,('#2368b1','#c8751a')):
  curve=report['versions'][version]['threshold_curve'];label='v1 conservative' if version==VERSIONS[0] else 'v2 recall calibrated'
  visible=[row for row in curve[::-1] if row['proposals']>0]
  axes[0].plot([r['recall'] for r in visible],[r['precision'] for r in visible],'.-',label=label,color=color)
  primary=report['versions'][version]['at_0_6'];axes[0].scatter([primary['recall']],[primary['precision']],s=65,color=color,zorder=3)
  axes[1].plot([r['confidence_threshold'] for r in curve],[r['f2'] for r in curve],'.-',label=label,color=color)
 normalized=report['lossless_aliases_then_strict']['versions'][VERSIONS[1]]['threshold_curve']
 normalized_visible=[row for row in normalized[::-1] if row['proposals']>0]
 axes[0].plot([r['recall'] for r in normalized_visible],[r['precision'] for r in normalized_visible],'--',label='v2 + lossless aliases (secondary)',color='#8a4e19')
 axes[1].plot([r['confidence_threshold'] for r in normalized],[r['f2'] for r in normalized],'--',label='v2 + lossless aliases (secondary)',color='#8a4e19')
 axes[0].set(xlabel='Recall against reviewed boxes',ylabel='Precision against reviewed boxes',title='Precision–recall',xlim=(0,1.02),ylim=(0,1.02))
 axes[1].set(xlabel='Confidence threshold',ylabel='F2',title='Threshold sensitivity',xlim=(0,1),ylim=(0,1.02));axes[1].axvline(.6,color='#6a7781',linestyle='--',linewidth=1)
 for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=8)
 figure.suptitle(f"Zone 7 development · {report['paired_images']} positive images · box IoU ≥ 0.5",fontsize=11);figure.tight_layout()
 figure.savefig(output/'precision-recall.png',dpi=180);figure.savefig(output/'precision-recall.svg');plt.close(figure)
 report['analysis_revision']='schema-and-lower-confidence-v2';report['analysis_source_sha256']=file_hash(Path(__file__));save(output/'report.json',report)
 lines=['# Reviewed-image prompt comparison','',f"Measured development winner at confidence 0.6: **{report['measured_winner_at_0_6'] or 'tie'}**.",'',
  'The exact Robotics ER2 model saw the same 88 zone 7 development images under two frozen prompts. Zones 8 and 9 were not used. Reviewed boxes were available only to the evaluator.','',
  '| Prompt | Precision | Recall | F2 | Unmatched boxes/image |','|---|---:|---:|---:|---:|']
 for version in VERSIONS:
  s=report['versions'][version]['at_0_6'];lines.append(f"| {version} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f2']:.4f} | {s['unmatched_boxes_per_image']:.4f} |")
 lines+=['',report['interpretation'],'','## Lower-confidence proposals','']
 for version in VERSIONS:
  item=report['versions'][version]['lower_confidence_analysis'];lines.append(f"- {version}: retaining all lower-confidence proposals adds {item['additional_matched_boxes_below_0_6']} matched boxes and {item['additional_unmatched_boxes_below_0_6']} unmatched boxes. The accepted-training threshold remains 0.6.")
 lines+=['','## Production schema diagnostics','',
  '| Policy | V2 rejected frames | V2 matches at ≥0.6 | V2 matches at all confidences | V2 unmatched at all confidences |',
  '|---|---:|---:|---:|---:|']
 for key in ('strict_production_raw','lossless_aliases_then_strict'):
  policy=report[key];values=policy['versions'][VERSIONS[1]];low=values['threshold_curve'][0]
  lines.append(f"| {policy['policy']} | {len(policy['whole_image_rejected_frames'][VERSIONS[1]])} | {values['at_0_6']['tp']} | {low['tp']} | {low['unmatched_boxes']} |")
 lines+=['','The original raw-output comparison is unchanged. Lossless aliases rename only explicit xmin/ymin/xmax/ymax keys; values and confidence stay identical. An ambiguous `max` key remains invalid. Whole-image rejection discards every proposal in an invalid frame. All six observed malformed frames contained only low-confidence proposals, so the ≥0.6 accepted comparison is unchanged.','',
  'V2 is the operational candidate-generation choice for the stated recall goal, with low-confidence proposals retained for review. V1 remains the marginal raw F2 winner at the fixed 0.6 acceptance threshold. This is a descriptive single-pass development comparison, not proof of a general improvement.']
 lines+=['','## Reproduction','',f"Run `python -m aerorozum.prompt_comparison --output '{output}'`. Resume uses saved local/GCS responses before paid inference. Recompute figures with `--report-only` without invoking a model.",'',
  'Frozen protocol, per-frame raw provider logs, parsed proposals, paired predictions, CSV threshold curves, plots and qualitative overlays are stored together. Green boxes are proposals ≥0.6; amber are lower-confidence proposals; cyan are reviewed annotations. Evaluation records are excluded from training.']
 (ROOT/'docs/prompt-comparison.md').write_text('\n'.join(lines)+'\n');(output/'report.md').write_text('\n'.join(lines)+'\n')
 if archive:
  from .cloud_cache import GCSRecordStore
  bucket=json.loads((ROOT/'configs/labelling.json').read_text())['gcs_bucket'];store=GCSRecordStore(bucket,prefix='robotics-er2-labels/prompt-evaluations/'+protocol['protocol_sha256'][:24])
  uri=store.put(digest(['analysis-report',protocol['protocol_sha256'],report['analysis_revision'],digest(report)]),report,'progress');save(output/'analysis-archive.json',{'report_uri':uri,'report_sha256':digest(report),'at':utcnow()})
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--plan-only',action='store_true');p.add_argument('--report-only',action='store_true');args=p.parse_args();output=Path(args.output or data_root()/'prompt-comparisons/zone7-v1-v2-er2')
 if args.plan_only:print(json.dumps(freeze_protocol(data_root(),output)));return
 if args.report_only:print(json.dumps(finalize_report(output)));return
 print(json.dumps(asyncio.run(run(output=output))),flush=True)

if __name__=='__main__':main()
