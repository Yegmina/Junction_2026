"""Visualize exported native DuMux radial states, without new physical claims."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from aerorozum.common import ROOT

def main():
    source=Path('/tmp/aerorozum-radial-animation.json')
    doc=json.loads(source.read_text());times=np.array(doc['times_s']);hours=times/3600
    radial=np.array(doc['r_m']);temp=np.array(doc['temperature_k']);moist=np.array(doc['moisture'])
    if len(times)<2 or np.any(np.diff(times)<=0) or not np.isfinite(temp).all():raise ValueError('Invalid native solver states')
    out=ROOT/'docs/presentation/assets';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.facecolor':'#091b18','axes.facecolor':'#102720','text.color':'#f2f0de','axes.labelcolor':'#b4c9bb','xtick.color':'#b4c9bb','ytick.color':'#b4c9bb','axes.edgecolor':'#42614e','font.family':'DejaVu Sans','font.size':12})
    x=np.linspace(-.5,.5,180);xx,yy=np.meshgrid(x,x);query=np.hypot(xx,yy)
    fields=np.array([np.interp(query,radial,row) for row in temp])-273.15
    contrast=fields-(temp[:,-1,None,None]-273.15)
    limit=max(.05,float(np.max(np.abs(contrast))));norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    chosen=[int(np.argmin(np.abs(hours-v))) for v in (0,6,12,18)]
    for kind,title,data,cmap,vmin,vmax in [('temperature','Surface temperature',fields,'inferno',float(fields.min()),float(fields.max())),('contrast','Temperature difference from outer radial boundary',contrast,'coolwarm',-limit,limit)]:
        fig,axes=plt.subplots(1,4,figsize=(19.2,7.2),layout='constrained')
        fig.suptitle('AEROROZUM  /  '+title+'\nActual coupled DuMux states · 24-hour qualification case',fontsize=22,x=.02,ha='left')
        for ax,ix in zip(axes,chosen):
            image=ax.imshow(data[ix],extent=[-.5,.5,-.5,.5],origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
            ax.set(title=f'{hours[ix]:.2f} simulated hours',xlabel='x (m)',ylabel='y (m)')
        cb=fig.colorbar(image,ax=axes,shrink=.7);cb.set_label('Surface °C' if kind=='temperature' else 'Difference (K)')
        fig.supxlabel('Homogeneous soil + single cylindrical inclusion. Experimental numerical output; field validation not performed.',fontsize=13)
        fig.savefig(out/f'physics-{kind}-day.png',dpi=120);plt.close(fig)
    fig=plt.figure(figsize=(16,9),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,.08],width_ratios=[1.05,1.05,1.15])
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[0,2]);footer=fig.add_subplot(grid[1,:]);footer.axis('off')
    absolute=a.imshow(fields[0],extent=[-.5,.5,-.5,.5],origin='lower',cmap='inferno',vmin=fields.min(),vmax=fields.max())
    difference=b.imshow(contrast[0],extent=[-.5,.5,-.5,.5],origin='lower',cmap='coolwarm',norm=norm)
    fig.colorbar(absolute,ax=a,shrink=.55,label='Surface °C');fig.colorbar(difference,ax=b,shrink=.55,label='Difference (K)')
    a.set(title='Computed surface temperature',xlabel='x (m)',ylabel='y (m)');b.set(title='Contrast against outer boundary',xlabel='x (m)',ylabel='y (m)')
    c.plot(hours,temp[:,0]-273.15,color='#c8f09b',lw=2,label='Centre');c.plot(hours,temp[:,-1]-273.15,color='#e7b477',lw=2,label='Outer radial boundary');c.set(xlabel='Simulated hours',ylabel='Surface °C',title='A full heating / cooling cycle');c.legend(loc='upper right',framealpha=.12,labelcolor='#f2f0de');c.grid(alpha=.15)
    cursor=c.axvline(0,color='#f2f0de',ls='--',lw=1)
    title=fig.suptitle('',fontsize=23,x=.02,ha='left')
    footer.text(.01,.75,'ACTUAL SOLVER OUTPUT  /  HOMOGENEOUS SOIL + ONE CYLINDRICAL INCLUSION',color='#c8f09b',fontsize=12)
    footer.text(.01,.25,'Experimental, numerical admission pending. Fixed colour scales. Not recorded terrain or a validated camera image.',color='#b4c9bb',fontsize=11)
    video=out/'physics-heating-cooling.mp4';working=video.with_name('physics-heating-cooling.rendering.mp4');command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','4','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-vf','fps=24','-pix_fmt','yuv420p','-movflags','+faststart',str(working)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for i in range(len(times)):
        absolute.set_data(fields[i]);difference.set_data(contrast[i]);cursor.set_xdata([hours[i],hours[i]])
        title.set_text(f'AEROROZUM  /  Coupled thermal physics                       {hours[i]:05.2f} / {hours[-1]:.0f} simulated hours')
        fig.canvas.draw();pixels=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
        process.stdin.write(pixels.tobytes())
        if i==int(np.argmax(np.max(np.abs(contrast),axis=(1,2)))):fig.savefig(out/'physics-simulation-frame.png',dpi=120)
    process.stdin.close()
    if process.wait():raise RuntimeError('Physics video encoding failed')
    working.replace(video)
    plt.close(fig)
    receipt={'source_case':doc['case'],'export_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_states':len(times),'duration_simulated_s':times[-1],'research_accepted_at_export':False,'rendering':'Native radial samples linearly interpolated at display coordinates, no extrapolation; fixed colour scales; native sample times displayed','radial_profile_reference':'Difference from last exported radial cell; not a target-absent counterfactual'}
    (ROOT/'docs/presentation/physics-video-provenance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
