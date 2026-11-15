"""Build separate RGB appearance and a DuMux-driven experimental thermal scene."""
import argparse,hashlib,json,math,random,shutil
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from PIL import Image

BASE=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl')
SCENE=BASE/'simulation/sensor-scene'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def interpolate(data,key,time_s):
    times=data['times_s'];j=max(1,min(len(times)-1,int(np.searchsorted(times,time_s))))
    if not times[0]<=time_s<=times[-1]:raise ValueError('No physics-time extrapolation')
    a=float((time_s-times[j-1])/(times[j]-times[j-1]));return data[key][j-1]*(1-a)+data[key][j]*a

def sample2d(array,x,y,x_axis,y_axis):
    ix=np.interp(x,x_axis,np.arange(len(x_axis)));iy=np.interp(y,y_axis,np.arange(len(y_axis)))
    x0=np.floor(ix).astype(int);y0=np.floor(iy).astype(int);x1=np.minimum(x0+1,len(x_axis)-1);y1=np.minimum(y0+1,len(y_axis)-1)
    wx=ix-x0;wy=iy-y0
    return (1-wx)*(1-wy)*array[y0,x0]+wx*(1-wy)*array[y0,x1]+(1-wx)*wy*array[y1,x0]+wx*wy*array[y1,x1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--physics-hour',type=float,default=7.5);args=parser.parse_args()
    SCENE.mkdir(parents=True,exist_ok=True)
    rgb_source=SCENE/'sources/aerial_grass_rock_diff_4k.jpg'
    with Image.open(rgb_source) as source:source.load();source=source.convert('RGB')
    rng=random.Random(20261114);n=4608;tile=880;stride=704
    total=np.zeros((n,n,3),np.float32);weights=np.zeros((n,n),np.float32)
    ramp=np.maximum(.001,np.sin(np.linspace(0,np.pi,tile))**2).astype('float32');window=ramp[:,None]*ramp[None,:]
    for y in range(0,n-tile+1,stride):
        for x in range(0,n-tile+1,stride):
            p=source.resize((tile,tile),Image.Resampling.LANCZOS).rotate(rng.choice([0,90,180,270]))
            if rng.random()<.5:p=p.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            total[y:y+tile,x:x+tile]+=np.asarray(p,dtype=np.float32)*window[:,:,None];weights[y:y+tile,x:x+tile]+=window
    rgb=np.clip(total[128:4224,128:4224]/weights[128:4224,128:4224,None],0,255).astype('uint8')
    rgb_path=SCENE/'rgb-ground.jpg';Image.fromarray(rgb).save(rgb_path,quality=94)
    del total,weights
    physical=BASE/'artifacts/physics';present_path=physical/'dumux-surface-day-present/field.npz';absent_path=physical/'dumux-surface-day-absent/field.npz'
    present=np.load(present_path);absent=np.load(absent_path)
    for key in ('x_m','y_m','times_s'):
        if not np.array_equal(present[key],absent[key]):raise ValueError('Unmatched DuMux counterfactual grids')
    t=args.physics_hour*3600;a=interpolate(present,'temperature_k',t);b=interpolate(absent,'temperature_k',t);delta=a-b
    texture=BASE/'simulation/models/recorded_soil/materials/textures/recorded_ground.png'
    with Image.open(texture) as im:appearance=np.asarray(im.convert('L').resize((4096,4096)),dtype=np.float32)
    appearance=(appearance-appearance.mean())/max(1,float(appearance.std()))
    # Reduced-order periodic soil heat diffusion: surface admittance is
    # h + (1+i)*sqrt(k*C*omega/2). Albedo is an assigned RGB material proxy,
    # not a calibrated material measurement. DuMux supplies the baseline and
    # buried-inclusion response; this couples both sensors to the same terrain.
    luminance=np.asarray(Image.fromarray(rgb).convert('L'),dtype=np.float32)/255
    albedo=.12+.25*luminance;omega=2*math.pi/86400
    mean_temperature=float(b.mean());h=12+4*.95*5.670374419e-8*mean_temperature**3
    admittance=h+(1+1j)*math.sqrt(.8*2e6*omega/2)
    transfer=float(np.real(np.exp(1j*omega*(t-21600))/admittance))
    material_response=-650*(albedo-float(albedo.mean()))*transfer
    field=np.full((4096,4096),mean_temperature,np.float32)+material_response+appearance*.06
    x_axis=np.linspace(-35,35,4096);y_axis=np.linspace(35,-35,4096)
    truth=json.loads((BASE/'simulation/provenance/scene.truth.json').read_text())
    for index,target in enumerate(truth['targets']):
        cx,cy=float(target['x_m']),float(target['y_m']);ix=np.where(abs(x_axis-cx)<=.5)[0];iy=np.where(abs(y_axis-cy)<=.5)[0]
        xx,yy=np.meshgrid(x_axis[ix]-cx+.5,y_axis[iy]-cy+.5)
        patch=sample2d(delta,xx,yy,present['x_m'],present['y_m'])
        # The physical domain boundary contrast is practically zero; a narrow
        # edge taper prevents a visible insertion seam in the composed scene.
        edge=np.minimum.reduce([xx,yy,1-xx,1-yy]);patch*=np.clip(edge/.08,0,1)
        field[np.ix_(iy,ix)]+=patch
    field_path=SCENE/'temperature_k.npy';np.save(field_path,field,allow_pickle=False)
    field_sha=sha(field_path);immutable=SCENE/'fields'/(field_sha+'.npy');immutable.parent.mkdir(exist_ok=True)
    if not immutable.exists():shutil.copy2(field_path,immutable)
    manifest=dict(version='paired-physics-camera-v2-material',physics_time_s=t,physics_time_hours=args.physics_hour,world_bounds=dict(x_min=-35,y_min=-35,x_max=35,y_max=35),array_rows='north_to_south',thermal_field_sha256=field_sha,thermal_field_path=str(immutable),temperature_range_k=[float(field.min()),float(field.max())],background_temperature_k=float(b.mean()),background_microtexture_sigma_k=.22,thermal_appearance_sha256=sha(texture),rgb_texture_sha256=sha(rgb_path),rgb_source=dict(url='https://polyhaven.com/a/aerial_grass_rock',download_url='https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/aerial_grass_rock/aerial_grass_rock_diff_4k.jpg',license='CC0',author='Rob Tuytel',sha256=sha(rgb_source),original_width_m=15),physics=dict(backend='dumux_extended_richards_ni',present_sha256=sha(present_path),absent_sha256=sha(absent_path),native_grid_shape=list(delta.shape),native_domain_m=[1,1],contrast_range_k=[float(delta.min()),float(delta.max())],method='Time-interpolated target-present minus target-absent surface field, composed at native metre scale into each buried-target location'),research_accepted=False,limitations=['The 70m scene is a composition of 1m DuMux solutions, not a full 70m coupled PDE solve.','Recorded thermal appearance is empirical and not calibrated radiometry.','The existing DuMux fields have not passed full numerical/field acceptance.','RGB material is a separate photographic source, not registered real-flight RGB/thermal data.','Ground is planar; buried objects are not exposed geometric discs.'])
    manifest['background_microtexture_sigma_k']=.06
    manifest['material_response']=dict(model='Periodic semi-infinite heat diffusion with convective/radiative surface admittance',albedo='Assigned .12 + .25 * photographic diffuse luminance; not measured',conductivity_w_m_k=.8,volumetric_heat_capacity_j_m3_k=2e6,linear_surface_exchange_w_m2_k=float(h),solar_harmonic_w_m2=650,period_s=86400,solar_peak_time_s=21600,temperature_response_range_k=[float(material_response.min()),float(material_response.max())])
    manifest['limitations'].append('RGB-derived albedo and reduced-order ground material response are assigned, not experimentally calibrated; lateral heat flow is not solved across the composed 70m scene.')
    (SCENE/'scene.json').write_text(json.dumps(manifest,indent=2))
    (immutable.with_suffix('.json')).write_text(json.dumps(manifest,indent=2))
    world_path=BASE/'simulation/worlds/aerorozum.sdf';backup=world_path.with_suffix('.before-physics-camera.sdf')
    if not backup.exists():shutil.copy2(world_path,backup)
    tree=E.parse(world_path);world=tree.getroot().find('world')
    for model in list(world.findall('model')):
        name=model.get('name','')
        if name.startswith(('inert_survey_target_','vegetation_')):world.remove(model)
    ground=world.find("model[@name='recorded_soil_ground']/link/visual")
    if ground is None:raise ValueError('Expected isolated recorded-soil ground')
    for plugin in list(ground.findall('plugin')):ground.remove(plugin)
    ground.find('material/pbr/metal/albedo_map').text=str(rgb_path)
    for sensor in list(world.findall("model[@name='aerorozum_drone']/link/sensor")):
        if sensor.get('type')=='thermal':
            for link in world.findall("model[@name='aerorozum_drone']/link"):
                if sensor in list(link):link.remove(sensor)
        elif sensor.get('name')=='rgb':sensor.find('pose').text='0 0 -.12 0 1.57079632679 0'
    E.indent(tree);tree.write(world_path,encoding='unicode',xml_declaration=True)
    print(json.dumps({k:manifest[k] for k in ('version','physics_time_hours','temperature_range_k','physics')},indent=2))

if __name__=='__main__':main()
