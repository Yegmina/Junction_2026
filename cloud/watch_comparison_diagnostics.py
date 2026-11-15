"""Publish read-only figures when predeclared comparison results settle."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cloud.export_expanded_comparison import export
from cloud.summarize_temporal_reviews import summarize

ROOT = Path('/opt/aerorozum')
ART = ROOT / 'artifacts'
NIGHT = ART / 'overnight-20261115T000942'
PUBLIC = ROOT / 'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'
WORK = NIGHT / 'presentation-work/expanded'
EVIDENCE = WORK / 'docs/evidence'
END = datetime.fromisoformat('2026-11-15T06:40:00+00:00')


def write(path, value):
    temporary = path.with_name('.' + path.name + '.next')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o644)
    temporary.replace(path)


def publish(paths, name):
    for path in paths:
        target = PUBLIC / path.name
        temporary = target.with_name('.' + target.name + '.next')
        shutil.copyfile(path, temporary)
        temporary.chmod(0o644)
        temporary.replace(target)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    subprocess.run(['/snap/bin/gcloud', 'storage', 'cp', *map(str, paths),
                    'gs://aerorozum-thermal-rl-avid-relic-509408-e4/research/overnight-20261115/'
                    'comparison-diagnostic-snapshots/' + name + '/' + stamp + '/', '--quiet'],
                   check=True, timeout=300)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    previous = None
    temporal_done = False
    while datetime.now(timezone.utc) < END:
        try:
            data = export(ART)
            signature = hashlib.sha256(json.dumps(data['source_sha256'], sort_keys=True).encode()).hexdigest()
            if signature != previous:
                write(EVIDENCE / 'expanded-comparison-v1.json', data)
                source = Path(__file__).resolve().parents[1]
                subprocess.run(['/opt/aerorozum/plot-venv/bin/python',
                                str(source / 'scripts/plot_expanded_comparison.py')],
                               cwd=WORK, check=True, timeout=120)
                publish([EVIDENCE / ('expanded-comparison-v1.' + ext)
                         for ext in ('json', 'png', 'pdf', 'svg')], 'expanded')
                previous = signature
                print(json.dumps(dict(at=data['at'], expanded_all_completed=data['all_stages_completed'])), flush=True)
            temporal = NIGHT / 'post-label-temporal-v1'
            if not temporal_done and (temporal / 'report.json').exists():
                result = summarize(temporal)
                path = EVIDENCE / 'temporal-review-diagnostic-v1.json'
                write(path, result)
                publish([path], 'temporal')
                temporal_done = True
                print(json.dumps(dict(at=result['at'], valid_temporal_pairs=result['valid_paired_frames'])), flush=True)
            if data['all_stages_completed'] and temporal_done:
                break
        except Exception as error:
            print(type(error).__name__ + ': ' + str(error)[:1000], flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
