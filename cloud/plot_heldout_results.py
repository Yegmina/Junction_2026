"""Plot completed, frozen held-out measurements without selecting checkpoints."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());by_model={r['model']:r for r in data['held_out_perception']}
    order=[('gemma-base','Gemma base'),('gemma-parent','Gemma parent'),('gemma','Gemma +96 GRPO · selected'),('gemini-robotics-er-2-preview','Robotics teacher'),('gemini-4-carbon','Gemini Carbon'),('gemini-4-argon','Gemini Argon')]
    if any(k not in by_model for k,n in order):raise ValueError('All six frozen comparisons must finish before plotting')
    rows=[by_model[k] for k,n in order];assert all(r['examples']==182 for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#edf2ec','axes.labelcolor':'#b0c4ba','xtick.color':'#b0c4ba','ytick.color':'#edf2ec','font.size':12})
    fig,(ax,errors)=plt.subplots(1,2,figsize=(16,9),facecolor='#101914',gridspec_kw={'width_ratios':[1.5,1]});y=np.arange(len(rows))
    for a in [ax,errors]:
        a.set_facecolor('#101914');a.invert_yaxis();a.tick_params(length=0,pad=10)
        for spine in a.spines.values():spine.set_visible(False)
    colors=['#71878d','#72e8c0','#b9f69a','#e5bc82','#e5bc82','#e5bc82']
    values=[100*r['micro_f2'] for r in rows];ax.barh(y,values,height=.52,color=colors);ax.set_yticks(y,[n for k,n in order]);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);ax.set_xlabel('Micro-F2 at IoU ≥ 0.5',labelpad=14);ax.grid(axis='x',color='#334037',alpha=.6);ax.set_axisbelow(True)
    for i,v in enumerate(values):ax.text(v+1.1,i,f'{v:.1f}%',va='center',weight='bold')
    errors.barh(y-.14,[r['fp'] for r in rows],height=.25,color='#d49182',label='False box claims');errors.barh(y+.14,[r['fn'] for r in rows],height=.25,color='#71878d',label='Missed reference boxes');errors.set_xlim(0,225);errors.set_yticks(y,[]);errors.set_xlabel('Counts (lower is better)',labelpad=14)
    for i,r in enumerate(rows):
        for key,delta in [('fp',-.14),('fn',.14)]:errors.text(r[key]+2,i+delta,str(r[key]),va='center',fontsize=10)
    legend=errors.legend(loc='lower center',bbox_to_anchor=(.5,1.025),frameon=False,fontsize=11)
    for t in legend.get_texts():t.set_color('#edf2ec')
    fig.text(.045,.94,'AEROROZUM / FROZEN HELD-OUT IMAGE COMPARISON',color='#b9f69a',fontsize=13,weight='bold');fig.text(.045,.87,'Measure the transfer. Keep the limits visible.',fontsize=28,weight='bold');fig.text(.045,.82,'182 previously held-out images · 208 reference boxes · checkpoint selected before test opening',color='#b0c4ba',fontsize=13)
    fig.subplots_adjust(left=.27,right=.94,top=.74,bottom=.25,wspace=.20)
    fig.text(.045,.105,'Mixed original-manual and reviewed-AI labels. Image agreement, not verified physical mine accuracy.',color='#b0c4ba',fontsize=11);fig.text(.045,.067,'Gemma: greedy 192 tokens. Gemini: native JSON, 8192 tokens including default reasoning. Not matched compute.',color='#b0c4ba',fontsize=11);fig.text(.045,.029,'Single-seed post-training study. No test score selected a model or changed training.',color='#b0c4ba',fontsize=11)
    fig.savefig(out/'heldout-comparison.png',dpi=120,facecolor=fig.get_facecolor());fig.savefig(out/'heldout-comparison.svg',facecolor=fig.get_facecolor(),metadata={'Date':None});plt.close(fig)
    (out/'heldout-comparison-provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frozen_selection_sha256=data['held_out_selection_sha256']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(a.source,a.output)
