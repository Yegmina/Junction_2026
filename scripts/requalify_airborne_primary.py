"""Policy-v2 qualification from saved original joins; no model requests."""
import copy,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from aerorozum.common import data_root,save,utcnow
from aerorozum.flight_filter import POLICY
from aerorozum.labelling import MODEL
r=data_root();out=r/'pipeline/inputs';auditfile=out/'airborne-eligibility.json';oldraw=auditfile.read_bytes();audit=json.loads(oldraw);oldsha=hashlib.sha256(oldraw).hexdigest();history=out/'airborne-policy-history'/oldsha;history.mkdir(parents=True,exist_ok=True)
if not (history/'airborne-eligibility.json').exists():(history/'airborne-eligibility.json').write_bytes(oldraw)
rows={row['id']:row for p in (r/'labelling').glob('*/frames/*/sampling.json') for row in json.loads(p.read_text()).get('images',[])};prior={}
for channel in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne'):
 for path in (r/'labelling'/channel/'labels/results').glob('*.json'):
  paid=json.loads(path.read_text())
  if paid.get('model')==MODEL and paid.get('sha256'):prior[(paid['sha256'],str(paid.get('source',{}).get('generation')))]= {'status':'invalid_paid_output_retained' if paid.get('status')=='invalid_model_response' else 'prior_paid_result_preserved','cache_key':paid.get('cache_key',path.stem)}
previous_policy=audit['policy']['version'];eligible=[];counts=Counter();reasons=Counter();per_source=defaultdict(Counter)
for f in audit['frames']:
 e=f['eligibility'];e['policy_version']=POLICY['version']
 if e.get('landed_state')==2 and e.get('thermal_receipt_ns') and e.get('telemetry') and 0<=e.get('state_age_s',float('inf'))<=2:
  e.update(status='eligible',airborne_eligible=True,reason='fresh_unambiguous_autopilot_in_air',relative_altitude_reference='HOME, not height above ground',altitude_diagnostic_status='unavailable' if 'relative_altitude_m' not in e else 'fresh' if 0<=e.get('altitude_age_s',float('inf'))<=2 else 'stale')
 if e['status']=='eligible':
  row=copy.deepcopy(rows[f['id']]);path=Path(row['image']);path=path if path.is_absolute() else r/path
  assert path.is_file() and path.resolve().is_relative_to(r.resolve()) and row['sha256']==f['sha256']
  row['image']=str(path.relative_to(r));row.update(airborne_eligible=True,eligibility=e,prior_paid_inference=prior.get((row['sha256'],str(row['source']['generation'])),{'status':'not_found_in_canonical_paid_cache'}));eligible.append(row)
 counts[e['status']]+=1;reasons[e['reason']]+=1;per_source[f['source']['object']][e['status']]+=1
pending=[x for x in eligible if (x['sha256'],str(x['source']['generation'])) not in prior];summary={'total_frames':len(audit['frames']),'airborne_eligible':len(eligible),'ground_excluded':counts['ground'],'unknown_skipped':counts['unknown'],'airborne_with_prior_paid_cache':len(eligible)-len(pending),'airborne_pending_frames':len(pending),'estimated_new_unique_inference_requests':len({(x['sha256'],str(x['source']['generation'])) for x in pending})}
audit.update(created_at=utcnow(),policy=POLICY,counts=summary,reasons=dict(reasons),per_source={k:dict(v) for k,v in per_source.items()},requalified_from={'audit_sha256':oldsha,'audit':str((history/'airborne-eligibility.json').relative_to(r)),'policy_version':previous_policy,'reason':'Relative altitude is HOME-relative, never an AGL threshold.'});save(auditfile,audit);sha=hashlib.sha256(auditfile.read_bytes()).hexdigest();snapshot=out/'airborne-snapshots'/sha[:16];save(snapshot/'airborne-eligibility.json',audit)
manifest={'schema_version':1,'status':'airborne_only','created_at':audit['created_at'],'policy':POLICY,'eligibility_manifest_sha256':sha,'eligibility_manifest':str((snapshot/'airborne-eligibility.json').relative_to(r)),'coverage_status':'complete_for_frozen_source_snapshot','source_versions_processed':len(audit['sources']),'source_versions_total':len(audit['sources']),'counts':summary,'session_splits':json.loads((r/'labelling/robotics-er2-5fps/session-splits.json').read_text())['sessions'],'images':eligible,'previous_paid_labels_preserved':True};save(snapshot/'airborne-eligible.json',manifest);save(out/'airborne-eligible.json',manifest)
fullsha=hashlib.sha256((snapshot/'airborne-eligible.json').read_bytes()).hexdigest();save(snapshot/'airborne-pending.json',{**manifest,'images':pending,'subset':'pending_only','full_manifest':str((snapshot/'airborne-eligible.json').relative_to(r)),'full_manifest_sha256':fullsha});state={'status':'complete_for_frozen_source_snapshot','at':utcnow(),'sources_processed':len(audit['sources']),'sources_total':len(audit['sources']),'policy_version':POLICY['version'],'counts':summary,'snapshot':str((snapshot/'airborne-eligible.json').relative_to(r)),'pending_snapshot':str((snapshot/'airborne-pending.json').relative_to(r)),'paid_inference_issued':False};save(r/'labelling/flight-filter/progress.json',state);print(json.dumps(state))
