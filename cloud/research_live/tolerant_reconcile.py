"""Re-score stored and arriving experiment outputs with Aki-box tolerance."""
import argparse,csv,json,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from score_tolerant import score

def atomic(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run(root,interval=10):
 root=Path(root);protocol=json.loads((root/'protocol.json').read_text());addendum=dict(version='aki-box-tolerant-v3',at=datetime.now(timezone.utc).isoformat(),source='User direction',rule='one-to-one match if IoU >= 0.5 OR every predicted edge is within the Aki box expanded by 35% of its width horizontally and 35% of its height vertically; small fully contained boxes count regardless of IoU; predictions are evaluated at confidence >= 0.6',confidence_threshold=.6,iou_threshold=.5,edge_tolerance_fraction=.35,one_to_one=True,original_iou_scores_preserved=True,heldout_test_accessed=False,training_eligible=False)
 atomic(root/'scoring-addendum.json',addendum);catalog={r['id']:r for r in protocol['images']};out=root/'tolerant-comparison.json'
 while True:
  groups=defaultdict(list);errors=[];records=0
  for path in (root/'results').glob('*.json'):
   try:
    row=json.loads(path.read_text());gold=catalog[row['id']]['boxes'];new=score(row.get('predictions') or [],gold);row['score_tolerant']=new;row['scoring_rule_version']=addendum['version'];atomic(path,row);groups[(row['prompt_version'],row['thinking_level'])].append(row);records+=1
   except (OSError,ValueError,KeyError,TypeError) as e:errors.append(path.name)
  arms=[]
  for (prompt,level),rows in sorted(groups.items()):
   tp=sum(r['score_tolerant']['tp'] for r in rows);fp=sum(r['score_tolerant']['fp'] for r in rows);fn=sum(r['score_tolerant']['fn'] for r in rows);arms.append(dict(prompt=prompt,thinking_level=level,completed=len(rows),expected=len(catalog)*protocol['replicas'],tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f2=5*tp/max(1,5*tp+4*fn+fp),invalid=sum(r.get('status')=='invalid_paid_response' for r in rows),transport_errors=sum(r.get('status')=='transport_error' for r in rows)))
  summary=dict(status='completed' if records==len(catalog)*len(protocol['prompts'])*len(protocol['thinking_levels'])*protocol['replicas'] and not errors else 'running',updated_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=protocol['sha256'],scoring_addendum=addendum,images=len(catalog),annotation_origins=protocol.get('annotation_origin_counts',{}),expected=len(catalog)*len(protocol['prompts'])*len(protocol['thinking_levels'])*protocol['replicas'],completed=records,arms=arms,unreadable_results=errors[:50],training_eligible=False,heldout_test_accessed=False)
  atomic(out,summary)
  if summary['status']=='completed':
   with (root/'tolerant-comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['prompt','thinking_level','completed','expected','tp','fp','fn','precision','recall','f2','invalid','transport_errors']);writer.writeheader()
    for row in arms:writer.writerow(dict(prompt=row['prompt'],**{k:v for k,v in row.items() if k!='prompt'}))
   return
  time.sleep(interval)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
