"""Export aggregate measured comparisons; raw outputs remain in private GCS."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import statistics


def f2(rows):
    tp=sum(r['score']['tp'] for r in rows);fp=sum(r['score']['fp'] for r in rows);fn=sum(r['score']['fn'] for r in rows)
    return dict(tp=tp,fp=fp,fn=fn,micro_f2=5*tp/max(1,5*tp+4*fn+fp))


def main():
    from aerorozum.common import save,utcnow
    root=Path('/opt/aerorozum/artifacts/overnight-20261115T000942/robotics-reasoning-v1')
    protocol=json.loads((root/'protocol.json').read_text());report=json.loads((root/'report.json').read_text())
    if report['status']!='completed':raise ValueError('Do not summarize an incomplete paired study as complete')
    source={r['id']:r for r in protocol['images']};groups=defaultdict(list)
    for path in (root/'results').glob('*.json'):
        row=json.loads(path.read_text());groups[(row['prompt_version'],row['thinking_level'],row['replica'])].append(row)
    arms=[]
    for (prompt,level,replica),rows in sorted(groups.items()):
        original=[r for r in rows if source[r['id']].get('provenance')=='original_manual']
        archived=[r for r in rows if source[r['id']].get('zone')=='zone7']
        api_times=[]
        for r in rows:
            identity=(r.get('raw_response') or {}).get('id');path=root/'provider'/level/'responses'/(str(identity)+'.json')
            if path.exists():api_times.append(json.loads(path.read_text())['elapsed_s'])
        arms.append(dict(prompt_version=prompt,thinking_level=level,replica=replica,
            all_development=f2(rows),archived_references=f2(archived),original_manual_only=f2(original),
            valid_outputs=sum(r.get('valid_output',False) for r in rows),images=len(rows),
            mean_rpc_elapsed_s=statistics.mean(api_times) if api_times else None,
            rpc_elapsed_samples=len(api_times),queue_inclusive_mean_elapsed_s=statistics.mean(r['elapsed_s'] for r in rows),
            thought_tokens=sum((r.get('usage',{}).get('thoughts_token_count') or 0) for r in rows),
            prompt_tokens=sum((r.get('usage',{}).get('prompt_token_count') or 0) for r in rows),
            answer_tokens=sum((r.get('usage',{}).get('candidates_token_count') or 0) for r in rows)))
    summary=dict(at=utcnow(),status='completed',protocol_sha256=protocol['sha256'],arms=arms,
        model=protocol['model'],max_output_tokens=protocol['max_output_tokens'],
        per_replica_reference_provenance=dict(Counter(r.get('provenance') for r in protocol['images'])),
        requests=2160,first_completion_single_attempt=True,transport_failures=0,
        inference_caps_equal_across_arms=True,test_accessed=False,training_eligible=False,
        rpc_elapsed_definition='Provider dispatch to received/parsed response; excludes quota admission wait',
        error_bar_definition='Standard deviation across three generation replicas; not a population generalization confidence interval',
        interpretation='Mixed original-manual/reviewed-AI development label agreement. Machine-screened negatives do not prove absence.')
    save(root/'aggregate-comparison.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
