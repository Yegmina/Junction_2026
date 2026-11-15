"""Render actual completed local 3D fields without modifying physics artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    from aerorozum.common import save,utcnow
    root=Path('/opt/aerorozum');case=root/'physics/cases/three-d-correspondence-v2/object-h4-local'
    out=root/'artifacts/overnight-20261115T000942/physics-full-day-diagnostic-v1';out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((case/'field.json').read_text());data=np.load(case/'field.npz')
    t=data['times_s'];x=data['x_m'];y=data['y_m'];temperature=data['temperature_k'].astype(float)
    if t[0]!=0 or t[-1]<86400 or np.any(np.diff(t)<=0) or not np.isfinite(temperature).all():raise ValueError('Expected complete finite monotone full-day field')
    xx,yy=np.meshgrid(x,y);radius=np.hypot(xx-.5,yy-.5)
    center=temperature[:,radius<=.08].mean(axis=1);background=np.median(temperature[:,(radius>=.3)&(radius<=.4)],axis=1)
    indices=[int(np.argmin(abs(t-v))) for v in [6*3600,12*3600,18*3600,24*3600]]
    anomaly=temperature-background[:,None,None];limit=max(float(np.max(np.abs(anomaly[i]))) for i in indices)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,8),layout='constrained');grid=fig.add_gridspec(3,4)
    axes=[fig.add_subplot(grid[:2,i]) for i in range(4)]
    for ax,index in zip(axes,indices):
        field=ax.pcolormesh(x,y,anomaly[index],cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
        hour=(metadata.get('forcing_start_hour',6)+t[index]/3600)%24
        ax.set_title(f'{int(hour):02d}:{int(round((hour%1)*60)):02d} forcing clock\n+{t[index]/3600:.1f} physical hours',fontsize=10)
        ax.set_aspect('equal');ax.set_xlabel('East, m')
    axes[0].set_ylabel('North, m');fig.colorbar(field,ax=axes,label='Surface temperature minus surrounding-ring median, K',shrink=.65)
    a=fig.add_subplot(grid[2,:2]);a.plot(t/3600,center-273.15,label='Center disk mean');a.plot(t/3600,background-273.15,label='Surrounding ring median');a.set_ylabel('Surface temperature, °C');a.legend(frameon=False,fontsize=9)
    b=fig.add_subplot(grid[2,2:]);b.plot(t/3600,center-background,color='#087e8b');b.axhline(0,color='gray',lw=.7);b.set_ylabel('Center − ring, K')
    for ax in [a,b]:ax.set_xlabel('Elapsed physical time, hours');ax.grid(alpha=.2);ax.set_xlim(0,24)
    fig.suptitle('A completed physical day · local 3D numerical diagnostic',fontsize=19,fontweight='bold',x=.02,ha='left')
    fig.supxlabel('Actual DuMux 3.10.0 solved states · homogeneous soil with one modeled cylinder · 2,887 adaptive surface samples\nMass/energy and interface checks passed. Independent grid/time/domain qualification and outdoor calibration remain separate.\nThis completed 3D case is unaccepted for research training. Ambient subtraction shows spatial contrast; no camera response or photographic terrain is implied.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig(out/('physics-full-day-diagnostic-v1.'+suffix),dpi=160)
    receipt=dict(at=utcnow(),source_field=str(case/'field.npz'),source_sha256=hashlib.sha256((case/'field.npz').read_bytes()).hexdigest(),
        source_metadata_sha256=hashlib.sha256((case/'field.json').read_bytes()).hexdigest(),
        source_research_accepted=metadata['research_accepted'],physical_duration_s=float(t[-1]),actual_surface_samples=len(t),
        displayed_actual_times_s=[float(t[i]) for i in indices],center_disk_radius_m=.08,background_ring_m=[.3,.4],
        background_statistic='per-state ring median',reference_is_counterfactual=False,unchanged_source_files=True,
        center_minus_ring_min_k=float((center-background).min()),center_minus_ring_max_k=float((center-background).max()))
    save(out/'receipt.json',receipt)
    # Movie frames interpolate between actual adaptive solved states in time.
    movie=out/'movie-frames';movie.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');image=ax.imshow(anomaly[0],origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest');fig.colorbar(image,ax=ax,label='Surface − surrounding ring, K');ax.set_xlabel('East, m');ax.set_ylabel('North, m')
    fig.supxlabel('Actual local 3D solver states, linearly interpolated in physical time\nHomogeneous local geometry · numerical admission pending · no camera calibration',fontsize=9)
    for frame,instant in enumerate(np.linspace(t[0],t[-1],96)):
        right=min(len(t)-1,max(1,int(np.searchsorted(t,instant))));left=right-1;weight=(instant-t[left])/(t[right]-t[left])
        image.set_data((1-weight)*anomaly[left]+weight*anomaly[right]);ax.set_title(f'Physical day: +{instant/3600:05.2f} h · DuMux 3D',fontweight='bold')
        fig.savefig(movie/f'{frame:04}.png',dpi=120)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','8','-i',str(movie/'%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','22','-threads','2','-movflags','+faststart',str(out/'physics-full-day-diagnostic-v1.mp4')],check=True)
    receipt.update(movie_frames=96,movie_fps=8,temporal_interpolation='linear between actual solved states; no extrapolation')
    save(out/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
