"""Non-radiometric camera appearance statistics from unlabelled real footage."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from .common import data_root,save,utcnow


def calibrate(directory, maximum=32):
    paths=sorted(p for p in Path(directory).glob('frame_*.jpg') if p.stem.removeprefix('frame_').isdigit())
    if not paths:raise ValueError('No original native thermal frames found')
    indices=np.linspace(0,len(paths)-1,min(maximum,len(paths))).astype(int)
    spectra=[];contrasts=[];noise=[];medians=[];sources=[]
    size=256
    fy=np.fft.fftfreq(size)[:,None];fx=np.fft.rfftfreq(size)[None,:]
    radius=np.sqrt(fx*fx+fy*fy);edges=np.linspace(0,.5,65)
    for index in indices:
        path=paths[index];image=Image.open(path)
        if image.size!=(640,512):continue
        a=np.asarray(image.convert('L').resize((size,size)),dtype=float)/255
        medians.append(float(np.median(a)))
        a-=a.mean();window=np.hanning(size)[:,None]*np.hanning(size)[None,:]
        power=np.abs(np.fft.rfft2(a*window))**2
        spectrum=np.array([np.mean(power[(radius>=lo)&(radius<hi)]) for lo,hi in zip(edges[:-1],edges[1:])])
        spectrum/=max(spectrum.sum(),1e-20);spectra.append(spectrum)
        contrasts.append(float(np.percentile(a,98)-np.percentile(a,2)))
        noise.append(float(np.median(np.abs(a[1:]-a[:-1]))))
        sources.append(dict(filename=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not spectra:raise ValueError('Calibration needs original 640x512 images without annotation overlays')
    d=dict(created_at=utcnow(),status='appearance_fitted_not_physics_validated',source_frames=len(sources),sources=sources,frequency_centers=((edges[:-1]+edges[1:])/2).tolist(),radial_power=np.median(spectra,axis=0).tolist(),display_contrast_p02_p98=float(np.median(contrasts)),display_median=float(np.median(medians)),adjacent_pixel_difference=float(np.median(noise)),radiometric=False,absolute_temperature_calibrated=False,limitations=['Pixels are contrast-adjusted, not measured temperature','Spatial scale depends on unknown height and nominal camera geometry','Source frames are unlabelled, not verified target-free'])
    save(data_root()/'appearance/profile.json',d)
    return d


def terrain(config,seed):
    """Random phase ground field with isotropic real-image spectrum, demo only.

    There are no deterministic wave components or repeated tiled photographs.
    Physical NPZ fields bypass this function entirely.
    """
    width=config['width_m'];height=config['height_m']
    nx=max(128,min(2048,int(width/.04)));ny=max(128,min(2048,int(height/.04)))
    rng=np.random.default_rng(seed)
    white=rng.normal(size=(ny,nx))
    fy=np.fft.fftfreq(ny)[:,None];fx=np.fft.rfftfreq(nx)[None,:]
    radius=np.sqrt(fx*fx+fy*fy)
    path=Path(config.get('appearance_profile_path',data_root()/'appearance/profile.json'))
    if path.is_file():
        import json
        d=json.loads(path.read_text())
        power=np.interp(radius,d['frequency_centers'],d['radial_power'])
        amplitude=np.sqrt(np.maximum(power,1e-12))
        source='real_frame_radial_spectrum_random_phase'
    else:
        amplitude=(1+(radius/.025)**2)**-.72
        source='seeded_multiscale_random_field'
    amplitude[0,0]=0
    field=np.fft.irfft2(np.fft.rfft2(white)*amplitude,s=white.shape)
    field=(field-field.mean())/max(field.std(),1e-10)
    return field.astype(np.float32),np.linspace(0,width,nx),np.linspace(0,height,ny),source
