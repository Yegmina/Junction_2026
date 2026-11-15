"""Recover receipt padding transparently; otherwise preserve video-relative time."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from .common import data_root,digest,save,utcnow
from .labelling import _probe,extract_video_samples

def receipt_records(path,cleaned_path):
 original=Path(path).read_bytes();cleaned=original.rstrip(b'\x00')
 records=[json.loads(line) for line in cleaned.splitlines() if line.strip()]
 if not records:raise ValueError('No valid original receipt records')
 Path(cleaned_path).parent.mkdir(parents=True,exist_ok=True);Path(cleaned_path).write_bytes(cleaned)
 return records,{'method':'remove_only_trailing_NUL_padding' if cleaned!=original else 'original_receipts_unchanged','original_sha256':hashlib.sha256(original).hexdigest(),'derived_sha256':hashlib.sha256(cleaned).hexdigest(),'trailing_NUL_bytes_removed':len(original)-len(cleaned),'timestamp_values_modified':False}

def prepare(root=None):
 root=Path(root or data_root());output=root/'labelling/quarantine-recovery';output.mkdir(parents=True,exist_ok=True)
 inputs=json.loads((root/'source-videos/labelling-sources.json').read_text())['videos'];splits=json.loads((root/'labelling/robotics-er2-5fps/session-splits.json').read_text())['sessions'];wanted=[s for s in inputs if (s['session_id']=='2026-11-04_161804' and s['segment'] in range(7)) or(s['session_id']=='2026-11-04_162621' and s['segment']==0)]
 rows=[];reports=[];clean_cache={}
 for source in wanted:
  width,height,pts=_probe(source['path'])
  if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Native video PTS not strictly monotonic; source remains quarantined')
  sid=source['session_id'];segment=source['segment'];raw=Path(source['timestamps_path']);derived=output/'receipts'/(sid+'.jsonl')
  if sid not in clean_cache:clean_cache[sid]=receipt_records(raw,derived)
  records,transform=clean_cache[sid];local=[r for r in records if int(r.get('segment',0))==segment];values=[int(r['received_at_ns']) for r in local]
  contiguous=all(int(b['frame'])==int(a['frame'])+1 for a,b in zip(local,local[1:]));aligned=len(local)==len(pts) and contiguous and all(b>a for a,b in zip(values,values[1:]))
  reason=None if aligned else 'receipt/native frame count mismatch' if len(local)!=len(pts) else 'nonmonotonic receipt clock or frame counter'
  info={'session_id':sid,'segment':segment,'native_frames':len(pts),'receipt_frames':len(local),'native_PTS_strictly_monotonic':True,'resolution':'original_receipts_recovered' if aligned else 'video_time_only','receipt_alignment_available':aligned,'GPS_alignment_available':False,'fallback_reason':reason,'receipt_transform':transform,'original_quarantine_retained':True,'original_failure':'Expecting value: line1 column1 at trailing NUL padding' if sid.endswith('161804') else 'Source timestamps must strictly increase'}
  provenance={k:source[k] for k in ('bucket','object','generation','session_id','segment')};provenance.update(timing_mode=info['resolution'],wallclock_alignment_available=aligned,gps_alignment_available=False,receipt_timestamp_recovery=info)
  folder=output/'frames'/digest(provenance)[:20];samples=extract_video_samples(source['path'],folder,session_id=sid,source=provenance,timestamps_path=derived if aligned else None,segment=segment,fps=5,split=splits[sid],sample_origin_s=int(records[0]['received_at_ns'])/1e9 if aligned else None)
  for row in samples:
   row.update(wallclock_alignment_available=aligned,gps_alignment_available=False,timing_resolution=info['resolution'])
   if aligned:row['receipt_timestamp_ns']=int(local[row['frame_index']]['received_at_ns'])
  rows.extend(samples);reports.append({**info,'sampled_frames':len(samples)})
 document={'metadata':{'status':'sampled_unlabelled','created_at':utcnow(),'model_not_called':True,'sampling_fps':5,'source_segments':len(reports),'receipt_aligned_segments':sum(x['receipt_alignment_available'] for x in reports),'video_time_only_segments':sum(not x['receipt_alignment_available'] for x in reports),'timestamp_values_fabricated':False,'GPS_alignment_available':False},'session_splits':splits,'sources':reports,'images':rows}
 save(output/'manifest.json',document);save(output/'timing-resolution.json',{'metadata':document['metadata'],'sources':reports});return document
