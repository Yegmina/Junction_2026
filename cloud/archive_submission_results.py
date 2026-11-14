"""Archive terminal experiment evidence to the existing private research bucket."""
import json
from pathlib import Path
import subprocess


def archive(root=Path('/opt/aerorozum'),gcloud='/snap/bin/gcloud'):
    data=root/'artifacts';bucket='gs://aerorozum-thermal-rl-avid-relic-509408-e4/'
    paths=set()
    for family in ['submission-reward-curriculum-v1','submission-visual-projection-v2',
                   'submission-refinement-v3','submission-visual-continuation-v1']:
        for file in (data/family).rglob('result.json'):
            result=json.loads(file.read_text())
            if result.get('status')!='completed':continue
            paths.add(file.parent)
            for stage in ['sft','grpo']:
                record=result.get('results',result).get(stage,{})
                if 'adapter' not in record:continue
                parent=(data/record['adapter']).resolve().parent
                if not parent.is_relative_to(data/'training'):raise ValueError('Adapter outside training tree')
                paths.add(parent)
    for folder in ['frontier-development-v1','pipeline/baseline-cache']:
        if (data/folder).is_dir():paths.add(data/folder)
    for path in sorted(paths):
        subprocess.run([gcloud,'storage','rsync','-r',str(path),bucket+'posttraining/'+str(path.relative_to(data))],check=True)
    for job in sorted((root/'harbor/jobs').iterdir()):
        result=job/'result.json'
        if not result.is_file() or not json.loads(result.read_text()).get('finished_at'):continue
        # OpenCode's cache may contain provider auth. Keep real run/trajectory
        # logs and verifier evidence; never copy the application cache subtree.
        subprocess.run([gcloud,'storage','rsync','-r','--exclude=(^|/)agent/opencode/.*',str(job),bucket+'harbor/jobs/'+job.name],check=True)


if __name__=='__main__':archive()
