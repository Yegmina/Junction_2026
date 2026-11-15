#!/usr/bin/env python3
"""Update request ETA from immutable source inventory and measured sample counts."""
import json,math,time
from collections import Counter
from datetime import datetime,timezone,timedelta
from aerorozum.common import data_root,save,utcnow
from aerorozum.labelling import MODEL,classify_response
r=data_root();out=r/'labelling';extra=out/'additional-bucket-sources';old=json.loads((r/'source-videos/labelling-sources-original-150.json').read_text())['videos'];objects={s['object'] for s in old};prior=json.loads((out/'eta-measurements.json').read_text());base=[dict(s) for s in prior['source_duration_estimates'] if s['object'] in objects];history=out/'eta-history/original-150-header-estimates.json'
if not history.exists():save(history,{'captured_at':utcnow(),'scope':'Original 150 immutable source objects','estimated_5fps_frames':sum(s['estimated_5fps_frames'] for s in base),'source_duration_estimates':base})
else:base=[dict(s) for s in json.loads(history.read_text())['source_duration_estimates']]
rows={s['object']:s for s in base}
for s in json.loads((extra/'old-zero-header-corrections.json').read_text())['sources']:
 rows[s['object']].update(estimated_5fps_frames=s['video_clock_5fps_frames'],duration_s=s['duration_s'],native_frames=s['native_frames'],count_basis='full_native_decode_monotonic_video_clock')
sampling=json.loads((out/'robotics-er2-5fps/sampling-manifest.json').read_text())
for s in sampling['sources']:
 if s.get('samples') is not None:rows[s['object']].update(estimated_5fps_frames=s['samples'],count_basis='actual_extracted_source_clock_5fps_samples')
q=json.loads((out/'quarantine-recovery/manifest.json').read_text());qc=Counter(row['source']['object'] for row in q['images'])
for name,count in qc.items():rows[name].update(estimated_5fps_frames=count,count_basis='actual_recovered_receipt_or_explicit_video_time_5fps_samples')
doc=json.loads((extra/'manifest.json').read_text())
for s in doc['sources']:
 if s['status']=='sampled':rows[s['object']]={'object':s['object'],'generation':s['generation'],'duration_s':s['duration_s'],'native_frames':s['native_frames'],'estimated_5fps_frames':s['sampled_frames'],'count_basis':'actual_extracted_source_clock_5fps_samples','timing_mode':s['timing']['mode']}
now=datetime.now(timezone.utc);valid=set();invalid=0
for folder in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne'):
 for path in (out/folder/'labels/results').glob('*.json'):
  result=json.loads(path.read_text());source=result.get('source',{})
  if result.get('model')!=MODEL or not source.get('object'):continue
  if result.get('status')=='invalid_model_response':invalid+=1;continue
  if result.get('frame_index') is None:continue
  try:
   if result.get('status')=='invalid_model_response':raise ValueError()
   classify_response(result['model_output'])
  except (ValueError,KeyError):invalid+=1;continue
  valid.add((source['object'],source.get('generation'),result['frame_index']))
recent=sum(path.stat().st_mtime>time.time()-600 for path in (out/'robotics-er2-5fps/labels/provider/responses').glob('*.json'));rate=recent/10
if not rate:rate=json.loads((out/'current-eta.json').read_text()).get('measured_requests_per_minute_10m',114)
total=sum(s['estimated_5fps_frames'] for s in rows.values());remaining=max(0,total-len(valid));minutes=math.ceil(remaining/rate);inventory=json.loads((extra/'generation-pinned-inventory.json').read_text());values=list(rows.values())
measure={**prior,'at':utcnow(),'videos':len(values),'normal_raw_objects':inventory['normal_count'],'recovered_duplicate_objects_excluded':1,'video_duration_hours':sum(float(s.get('duration_s') or 0) for s in values)/3600,'estimated_5fps_frames':total,'source_duration_estimates':values,'estimate_basis':'Original150 estimates corrected by full native decoding, progressively replaced with actual receipt-time samples; additional23 use actual extracted samples. Recovered435-frame clip is exact RGB+PTS duplicate and incurs no second inference.','source_inventory_status':f"{inventory['normal_count']} normal raw video objects + 1 audited recovered duplicate; derived annotated videos excluded",'new_sources_sampled':len(doc['sources']),'new_source_samples':len(doc['images']),'original_snapshot':'labelling/eta-history/original-150-header-estimates.json','measured_requests_per_minute_10m':rate}
unmeasured=max(0,len(inventory['new_normal_objects'])-len(doc['sources']))
measure['unmeasured_additional_source_versions']=unmeasured
save(out/'eta-measurements.json',measure);eta={'checked_at':utcnow(),'estimated_total_frames':total,'completed_native_source_frames':len(valid),'estimated_remaining_model_requests':remaining,'measured_requests_per_minute_10m':rate,'estimated_minutes_remaining':minutes,'projected_finish_utc':(now+timedelta(minutes=minutes)).isoformat(),'completed_percent':round(100*len(valid)/total,1),'invalid_paid_outputs_retained':invalid,'normal_raw_video_objects':inventory['normal_count'],'additional_source_sampled_frames':len(doc['images']),'recovered_raw_duplicate_excluded':True,'unmeasured_additional_source_versions':unmeasured,'estimate_is_lower_bound':bool(unmeasured),'limitations':['Original source estimates are progressively replaced by measured receipt-clock sample counts; totals can change.','New arrivals are checked every two minutes; bounded invalid-output retries are additional.','API-key remaining credit balance is not exposed by these local logs.']};save(out/'current-eta.json',eta);print(json.dumps(eta))
