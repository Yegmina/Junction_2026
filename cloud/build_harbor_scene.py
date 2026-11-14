"""Freeze a new private scene from a numerically admitted DuMux response pair.

This writes only a dedicated Harbor build's assets, never the live demo scene.
The reference positions are generated together with actual field insertions.
"""
import argparse,hashlib,json,math,random,shutil
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def interpolate(times,values,t):
    if not times[0]<=t<=times[-1]:raise ValueError('No physical time extrapolation')
    j=max(1,min(len(times)-1,int(np.searchsorted(times,t))));a=(t-times[j-1])/(times[j]-times[j-1])
    return values[j-1]*(1-a)+values[j]*a


def build(assets,case,seed=2026111403):
    assets=Path(assets).resolve();case=Path(case).resolve()
    if assets.name!='assets' or assets.parent.name!='backend':raise ValueError('Use the dedicated Harbor backend/assets build directory')
    certificate=json.loads((case/'admission-progress.json').read_text())
    if not certificate.get('accepted') or certificate.get('failed_checks') or any(c['status']!='passed' for c in certificate['checks']):raise ValueError('Numerical pair admission is incomplete')
    present=case/'present/field.npz';absent=case/'absent/field.npz'
    if sha(present)!=certificate['present_sha256'] or sha(absent)!=certificate['absent_sha256']:raise ValueError('Admitted source field hashes changed')
    with np.load(present,allow_pickle=False) as a,np.load(absent,allow_pickle=False) as b:
        times=a['times_s'];radii=a['r_m'];ta=a['radial_temperature_k'];tb=b['radial_temperature_k']
        if not np.array_equal(times,b['times_s']) or not np.array_equal(radii,b['r_m']):raise ValueError('Counterfactual sampling grids differ')
    rng=random.Random(seed);targets=[]
    for i,region in enumerate(['east','east','north']):
        for _ in range(1000):
            x,y=(rng.uniform(7,13),rng.uniform(-3,3)) if region=='east' else (rng.uniform(-3,3),rng.uniform(9,15))
            if all(math.hypot(x-t['x_m'],y-t['y_m'])>=3 for t in targets):break
        else:raise ValueError('Cannot separate inserted responses')
        targets.append(dict(id=f'inserted-{i}',x_m=x,y_m=y,radius_m=.12,depth_m=.03,material='effective_inert_polymer_surrogate'))
    scene=assets/'sensor-scene';n=4096
    rgb=np.asarray(Image.open(scene/'rgb-ground.jpg').convert('RGB').resize((n,n)),dtype=np.float32)/255
    luminance=.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2];del rgb
    texture=next((assets/'models').rglob('recorded_ground.png'))
    appearance=np.asarray(Image.open(texture).convert('L').resize((n,n)),dtype=np.float32)
    appearance=(appearance-appearance.mean())/max(1,float(appearance.std()))
    x_axis=np.linspace(-35,35,n);y_axis=np.linspace(35,-35,n);kept=set();phases={}
    for phase,t in [('warming',27000.),('cooling',61200.)]:
        source_t=interpolate(times,ta,t);absent_t=interpolate(times,tb,t);delta=source_t-absent_t;mean=float(absent_t.mean())
        if max(abs(delta))<.2:raise ValueError('Inserted contrast is below the frozen five-NETD visibility threshold')
        omega=2*math.pi/86400;exchange=12+4*.95*5.670374419e-8*mean**3
        transfer=float(np.real(np.exp(1j*omega*(t-21600))/(exchange+(1+1j)*math.sqrt(.8*2e6*omega/2))))
        albedo=.12+.25*luminance
        field=np.full((n,n),mean,np.float32)-650*(albedo-float(albedo.mean()))*transfer+appearance*.06
        for target in targets:
            ix=np.where(abs(x_axis-target['x_m'])<=radii[-1])[0];iy=np.where(abs(y_axis-target['y_m'])<=radii[-1])[0]
            xx,yy=np.meshgrid(x_axis[ix]-target['x_m'],y_axis[iy]-target['y_m']);distance=np.hypot(xx,yy)
            patch=np.interp(distance,radii,delta,left=delta[0],right=0.)
            field[np.ix_(iy,ix)]+=patch
        path=scene/'fields'/f'harbor-{phase}-{seed}.npy';np.save(path,field,allow_pickle=False);kept.add(path.name)
        prior=json.loads((scene/(phase+'-scene.json')).read_text())
        prior.update(version='harbor-admitted-axisymmetric-composition-v1',physics_time_s=t,physics_time_hours=t/3600,
            thermal_field_path='/scene/sensor-scene/fields/'+path.name,thermal_field_sha256=sha(path),background_temperature_k=mean,
            physics=dict(backend=certificate['backend'],present_sha256=certificate['present_sha256'],absent_sha256=certificate['absent_sha256'],
                pair_numerically_accepted=True,symmetry_scope=certificate['symmetry_scope'],field_validation='not_performed',
                contrast_range_k=[float(delta.min()),float(delta.max())],physical_day_s=86400,radial_samples=len(radii)),
            research_accepted=False,limitations=['Local axisymmetric response has numerical admission for homogeneous flat soil and uniform forcing.',
                'The 70m scene is a composition, not a full-domain heterogeneous PDE solve.',
                'Photographic RGB albedo and recorded thermal microtexture are assigned display appearance, not calibrated field measurements.',
                'Scores refer to explicitly inserted synthetic signatures; real mine detection is not validated.'])
        (scene/(phase+'-scene.json')).write_text(json.dumps(prior,indent=2));phases[phase]=prior
        np.savez_compressed(scene/(phase+'-response.npz'),r_m=radii,delta_k=delta,absent_k=absent_t,time_s=t)
    shutil.copyfile(scene/'warming-scene.json',scene/'scene.json')
    # This build directory is derived. The original deployed arrays are retained.
    for path in (scene/'fields').glob('*.npy'):
        if path.name not in kept:path.unlink()
    (assets/'provenance/scene.truth.json').write_text(json.dumps(dict(seed=seed,targets=targets,generated_together_with_insertions=True,
        field_hashes={phase:doc['thermal_field_sha256'] for phase,doc in phases.items()},reference_kind='simulator_private',frozen_before_run=True),indent=2))
    shutil.copyfile(case/'admission-progress.json',assets/'provenance/axisymmetric-admission.json')
    root=E.parse(assets/'worlds/aerorozum.sdf');world=root.getroot().find('world')
    for model in list(world.findall('model')):
        if model.get('name','').startswith('inert_survey_target_'):world.remove(model)
    root.write(assets/'worlds/aerorozum.sdf',encoding='unicode',xml_declaration=True)
    files=[dict(path=str(p.relative_to(assets)),sha256=sha(p),bytes=p.stat().st_size) for p in assets.rglob('*') if p.is_file()]
    receipt=dict(seed=seed,pair_admission=certificate['status'],composed_scene_field_validated=False,files=files)
    (assets.parent.parent.parent/'native-assets-manifest.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(target_count=len(targets),field_hashes={k:v['thermal_field_sha256'] for k,v in phases.items()},pair_numerically_accepted=True,whole_scene_field_validated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--case',required=True);p.add_argument('--seed',type=int,default=2026111403);a=p.parse_args();build(a.assets,a.case,a.seed)
