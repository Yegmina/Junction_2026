"""Describe paired temporal reviews without treating either machine arm as truth."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def summarize(folder: Path) -> dict:
    protocol_bytes = (folder / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    pairs = defaultdict(dict)
    invalid = Counter()
    for path in (folder / 'results').glob('*.json'):
        row = json.loads(path.read_text())
        assert row['protocol_sha256'] == protocol['sha256']
        assert row['training_eligible'] is False
        identity = (row['center_sha256'], str(row['source']['generation']), row['source']['object'])
        assert row['arm'] not in pairs[identity], 'Duplicate arm for a frozen frame identity'
        pairs[identity][row['arm']] = row
        if row['state'] != 'valid_model_review':
            invalid[row['arm']] += 1
    counts = Counter()
    by_source = defaultdict(Counter)
    accepted_totals = Counter()
    uncertain_totals = Counter()
    for identity, arms in pairs.items():
        if not all(arm in arms and arms[arm]['state'] == 'valid_model_review'
                   for arm in ('blind_context', 'proposal_context')):
            continue
        accepted = {arm: len(row['classification']['accepted']) for arm, row in arms.items()}
        category = ('accepted' if accepted['blind_context'] else 'none') + '/' + (
            'accepted' if accepted['proposal_context'] else 'none')
        counts[category] += 1
        by_source[identity[2]][category] += 1
        for arm, row in arms.items():
            accepted_totals[arm] += accepted[arm]
            uncertain_totals[arm] += len(row['classification']['uncertain'])
    report_path = folder / 'report.json'
    final = json.loads(report_path.read_text()) if report_path.exists() else None
    valid_pairs = sum(counts.values())
    result = dict(at=datetime.now(timezone.utc).isoformat(),
                  status='completed' if final and final['status'] == 'completed' else 'partial',
                  protocol_sha256=protocol['sha256'], protocol_file_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                  expected_pairs=len(protocol['cases']), observed_frame_identities=len(pairs),
                  valid_paired_frames=valid_pairs, invalid_outputs_by_arm=dict(invalid),
                  accepted_frame_table_blind_then_proposal=dict(counts),
                  accepted_candidate_totals=dict(accepted_totals), uncertain_candidate_totals=dict(uncertain_totals),
                  acceptance_disagreement_fraction=(counts['none/accepted'] + counts['accepted/none']) /
                  valid_pairs if valid_pairs else None,
                  by_source={k: dict(v) for k, v in sorted(by_source.items())},
                  provider_report=final, training_eligible=False, independent_accuracy_claimed=False,
                  interpretation='Descriptive same-provider paired review diagnostic, not recall, precision, or physical '
                  'confirmation. Both arms see the same three frames; the proposal arm additionally sees earlier '
                  'unverified detections. Acceptance uses the frozen 0.60 threshold. Invalid or missing arms are '
                  'excluded from the paired table and retained separately. No first-pass record is overwritten.')
    assert valid_pairs <= len(protocol['cases'])
    return result


if __name__ == '__main__':
    folder = Path('/opt/aerorozum/artifacts/overnight-20261115T000942/post-label-temporal-v1')
    print(json.dumps(summarize(folder), indent=2))
