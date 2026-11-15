"""Non-destructive 75% Aki-box rescore for past and arriving study records."""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from score_tolerant import score

VERSION = 'aki-box-tolerant-v4-75pct'
RULE = ('One-to-one match when IoU >= 0.5 OR all predicted box edges fall within '
        'the Aki reference box expanded by 75% of its width and height; '
        'confidence threshold 0.6. Original 35% records remain unchanged.')

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)

def study_outputs(root):
    patterns = (
        'private-prompt-study/*/output/protocol.json',
        'private-model-comparison/*/output/protocol.json',
        'private-height-prompt-study/*/output/protocol.json',
    )
    found = set()
    for pattern in patterns:
        for protocol_path in root.glob(pattern):
            found.add(protocol_path.parent)
    return sorted(found)

def rescore(output):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    audit_path = output / 'execution-audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get('exclude_from_comparison'):
            result = dict(version=VERSION, status=audit.get('status','invalid_setup'),
                scoring_rule=RULE, updated_at=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=protocol.get('sha256'), execution_audit=audit,
                completed=audit.get('completed',0), total=audit.get('total'), arms=[],
                historical_scores_preserved=True, heldout_test_accessed=False, training_eligible=False)
            atomic(output / 'tolerant-comparison-75.json',result)
            return result
    catalog = {row['id']: row for row in protocol.get('images', [])}
    groups = defaultdict(list)
    errors = []
    total_records = 0
    for path in (output / 'results').glob('*.json'):
        try:
            row = json.loads(path.read_text())
            if row.get('id') not in catalog:
                errors.append(path.name)
                continue
            model = row.get('model', protocol.get('model', 'unknown'))
            prompt = row.get('prompt_version', row.get('prompt', 'unknown'))
            level = row.get('thinking_level', 'unknown')
            item = dict(model=model, prompt=prompt, thinking_level=level, status=row.get('status'))
            if row.get('status') != 'transport_error':
                item['score'] = score(row.get('predictions') or [], catalog[row['id']].get('boxes') or [], edge_tolerance=.75)
            groups[(model, prompt, level)].append(item)
            total_records += 1
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(path.name)
    expected_per_arm = len(catalog) * int(protocol.get('replicas', 1))
    arms = []
    for (model, prompt, level), rows in sorted(groups.items()):
        scored = [row for row in rows if 'score' in row]
        tp = sum(row['score']['tp'] for row in scored)
        fp = sum(row['score']['fp'] for row in scored)
        fn = sum(row['score']['fn'] for row in scored)
        arms.append(dict(model=model, prompt=prompt, thinking_level=level,
            completed=len(rows), scored=len(scored), expected=expected_per_arm,
            tp=tp, fp=fp, fn=fn, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
            f2=5*tp/max(1,5*tp+4*fn+fp),
            invalid=sum(row['status']=='invalid_paid_response' for row in rows),
            transport_errors=sum(row['status']=='transport_error' for row in rows)))
    models=protocol.get('models') or ([protocol.get('model')] if protocol.get('model') else ['unknown'])
    expected = expected_per_arm * len(protocol.get('prompts', {})) * len(protocol.get('thinking_levels', [])) * len(models)
    transport_failures=sum(row['transport_errors'] for row in arms)
    status = ('completed_with_transport_errors' if transport_failures else 'completed') if total_records >= expected and not errors else 'running'
    result = dict(version=VERSION, scoring_rule=RULE, edge_tolerance_fraction=.75,
        confidence_threshold=.6, iou_threshold=.5, updated_at=datetime.now(timezone.utc).isoformat(),
        status=status, protocol_sha256=protocol.get('sha256'), baseline_protocol_sha256=protocol.get('base_protocol_sha256'),
        images=len(catalog), expected=expected, completed=total_records, transport_errors=transport_failures, arms=arms,
        unreadable_results=errors[:50], historical_scores_preserved=True,
        heldout_test_accessed=False, training_eligible=False)
    atomic(output / 'tolerant-comparison-75.json', result)
    csv_path = output / 'tolerant-comparison-75.csv'
    tmp = csv_path.with_suffix('.csv.tmp')
    fields = ['model','prompt','thinking_level','completed','scored','expected','tp','fp','fn','precision','recall','f2','invalid','transport_errors']
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(arms)
    tmp.replace(csv_path)
    return result

def run(root, interval=15):
    root = Path(root)
    while True:
        for output in study_outputs(root):
            try:
                rescore(output)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        time.sleep(interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/opt/aerorozum')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    run(args.root, args.interval)
