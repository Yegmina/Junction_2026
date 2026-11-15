"""Read-only export of every declared 2,400-image seed and continuation arm."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def export(root: Path) -> dict:
    files = {}
    rows = []
    fields = ('micro_f2', 'tp', 'fp', 'fn', 'valid_json_fraction',
              'negative_false_alarm_rate', 'examples', 'protocol_hash')
    for seed in (17, 43, 91):
        stages = []
        plans = [
            ('SFT', f'postsubmission-expanded-visual-v1/seed-{seed}/initial/result.json', 'sft'),
            ('Initial GRPO', f'postsubmission-expanded-visual-v1/seed-{seed}/initial/result.json', 'grpo'),
            ('Extra SFT', f'postsubmission-expanded-controls-v1/seed-{seed}/sft-control/initial/result.json', 'sft'),
            ('Extra GRPO', f'postsubmission-expanded-controls-v1/seed-{seed}/grpo/continuation/result.json', 'grpo'),
        ]
        for name, relative, kind in plans:
            path = root / relative
            row = dict(stage=name, state='pending', source=relative)
            if path.exists():
                raw = path.read_bytes()
                files[relative] = hashlib.sha256(raw).hexdigest()
                document = json.loads(raw)
                row['state'] = document['status']
                results = document.get('results', {})
                if document['status'] == 'completed':
                    evaluation = results[kind + '_evaluation']
                    training = results[kind]
                    assert evaluation['examples'] == 120
                    assert abs(evaluation['micro_f2'] - 5 * evaluation['tp'] /
                               max(5 * evaluation['tp'] + 4 * evaluation['fn'] + evaluation['fp'], 1)) < 1e-12
                    row.update({k: evaluation[k] for k in fields})
                    row.update(actual_updates=training['optimizer_updates'],
                               maximum_requested_updates=training['requested_optimizer_updates'],
                               early_stopped=training['early_stopped'],
                               early_stop_reason=training['early_stop_reason'],
                               train_runtime_s=training['metrics']['train_runtime'],
                               manifest_hash=training['manifest_hash'])
            stages.append(row)
        completed = [r for r in stages if r['state'] == 'completed']
        assert len({r['manifest_hash'] for r in completed}) <= 1
        rows.append(dict(seed=seed, stages=stages))
    return dict(at=datetime.now(timezone.utc).isoformat(), seeds=rows,
                all_stages_completed=all(s['state'] == 'completed' for r in rows for s in r['stages']),
                training_images=2400, development_images=120, source_sha256=files,
                development_reference_provenance=dict(original_manual=30, reviewed_ai=58, model_zero=32),
                maximum_continuation_updates=96, actual_update_matching_claimed=False,
                compute_matching_claimed=False, test_accessed=False, production_model_changed=False,
                interpretation='All declared seeds shown. Each continuation starts from its seed\'s exact initial GRPO parent. '
                'The guard can make actual updates differ. Reused development label agreement does not establish field '
                'generalization; model-zero references do not establish physical absence. Dataset size and training '
                'duration change together relative to the 602-image cohort.')


if __name__ == '__main__':
    print(json.dumps(export(Path('/opt/aerorozum/artifacts')), indent=2))
