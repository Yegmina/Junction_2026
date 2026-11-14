"""Stage a lossless, isolated native scene from the already deployed simulator."""
import hashlib,json,os,shutil
from pathlib import Path

base=Path('/opt/aerorozum/deployment/simulation')
task=Path(os.environ.get('AEROROZUM_HARBOR_SOURCE','/opt/aerorozum/harbor/source'))/'harbor/thermal-investigation'
backend=task/'environment/backend';assets=backend/'assets';assets.mkdir(parents=True,exist_ok=True)
for folder in ('models','worlds','provenance'):
 shutil.copytree(base/folder,assets/folder,dirs_exist_ok=True)
scene=assets/'sensor-scene';scene.mkdir(exist_ok=True)
shutil.copyfile(base/'sensor-scene/rgb-ground.jpg',scene/'rgb-ground.jpg')
for manifest in ('scene.json','warming-scene.json','cooling-scene.json'):
 doc=json.loads((base/'sensor-scene'/manifest).read_text());source=Path(doc['thermal_field_path']);dest=scene/'fields'/source.name;dest.parent.mkdir(exist_ok=True)
 if not dest.exists():shutil.copyfile(source,dest)
 assert hashlib.sha256(dest.read_bytes()).hexdigest()==doc['thermal_field_sha256']
 doc['thermal_field_path']='/scene/sensor-scene/fields/'+source.name
 (scene/manifest).write_text(json.dumps(doc,indent=2))
for path in assets.rglob('*'):
 if path.suffix in ('.sdf','.config','.material'):
  value=path.read_text();value=value.replace(str(base),'/scene');path.write_text(value)
runtime=backend/'runtime';runtime.mkdir(exist_ok=True)
repo=Path(__file__).resolve().parents[1]
if not (repo/'simulation/runtime.py').is_file():repo=Path('/opt/aerorozum/deployment/releases/coordinate-agent-v3')
for name in ('runtime.py','physics_camera.py','view_camera.py'):
 shutil.copyfile(repo/'simulation'/name,runtime/name)
shutil.copyfile(Path(__file__).with_name('build_harbor_scene.py'),runtime/'scene_factory.py')
shutil.copyfile(task/'environment/camera_geometry.py',runtime/'camera_geometry.py')
files=[dict(path=str(p.relative_to(assets)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in assets.rglob('*') if p.is_file()]
(task/'native-assets-manifest.json').write_text(json.dumps(dict(files=files,bytes=sum(p['bytes'] for p in files),lossless_fields=True,source=str(base),contains_original_recordings=False),indent=2))
print(json.dumps(dict(files=len(files),bytes=sum(p['bytes'] for p in files))))
