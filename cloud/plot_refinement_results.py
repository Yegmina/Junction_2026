"""Export the measured development comparison as standalone PNG and SVG."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());runs=data['runs'];initial=[r for r in runs if r['family'].endswith('v1')];warm=[r for r in runs if r['family'].endswith('v2')]
    initial.sort(key=lambda r:r['specification']['learning_rate']);warm.sort(key=lambda r:r['specification']['sft_steps']);runs=initial+warm
    names=[f'Initial SFT · LR {r["specification"]["learning_rate"]:g}' for r in initial]+[f'Warm start +{r["specification"]["sft_steps"]} updates' for r in warm]
    sft=[100*r['evaluations']['sft_evaluation']['micro_f2'] for r in runs];grpo=[100*r['evaluations']['grpo_evaluation']['micro_f2'] for r in runs]
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#eef5f4','axes.labelcolor':'#b0c4ca','xtick.color':'#b0c4ca','ytick.color':'#eef5f4','font.size':13})
    fig,ax=plt.subplots(figsize=(16,9),facecolor='#10191d');ax.set_facecolor('#10191d')
    y=np.arange(len(runs));ax.barh(y-.17,sft,height=.28,color='#71878d',label='SFT');ax.barh(y+.17,grpo,height=.28,color='#72e8c0',label='SFT + GRPO')
    for positions,values in [(y-.17,sft),(y+.17,grpo)]:
        for pos,value in zip(positions,values):ax.text(value+1,pos,f'{value:.1f}%',va='center',fontsize=12)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks(np.arange(0,101,20),[f'{n}%' for n in range(0,101,20)]);ax.set_xlabel('Micro-F2 at IoU ≥ 0.5',labelpad=14)
    ax.grid(axis='x',color='#30434a',alpha=.7);ax.set_axisbelow(True);ax.tick_params(axis='both',length=0,pad=12)
    for spine in ax.spines.values():spine.set_visible(False)
    legend=ax.legend(loc='lower right',frameon=False,fontsize=13);[t.set_color('#eef5f4') for t in legend.get_texts()]
    fig.text(.055,.94,'AeroRozum / GEMMA POST-TRAINING',color='#72e8c0',fontsize=14,weight='bold')
    fig.text(.055,.87,'Measured progress on thermal evidence.',fontsize=29,weight='bold')
    fig.text(.055,.82,'Six development experiments · identical evaluation images and greedy decoding',color='#b0c4ca',fontsize=14)
    fig.subplots_adjust(left=.27,right=.94,top=.75,bottom=.19)
    fig.text(.055,.095,'120 images: 88 reference objects + 32 screened negatives. Seed 17. Language-attention LoRA.',color='#b0c4ca',fontsize=12)
    fig.text(.055,.055,'Development selection; independent test and closed-loop frontier superiority remain unverified.',color='#b0c4ca',fontsize=12)
    fig.savefig(output/'refinement-results.png',dpi=120,facecolor=fig.get_facecolor());fig.savefig(output/'refinement-results.svg',facecolor=fig.get_facecolor(),metadata={'Date':None});plt.close(fig)
    (output/'refinement-results-provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sealed_test=False,frontier_comparison=False),indent=2))


def plot_latest(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());rows=[]
    for family,arm,stage,name in [
        ('submission-refinement-v3','base-control','base_control','Gemma base'),
        ('submission-refinement-v3','updates-480','grpo','Gemma language continuation + GRPO'),
        ('submission-visual-projection-v2','updates-480','sft','Gemma visual projection · SFT'),
        ('submission-visual-projection-v2','updates-480','grpo','Gemma visual projection · SFT + GRPO')]:
        match=next(r for r in data['other_training'] if r['family']==family and r['arm']==arm)
        rows.append((name,match['evaluations'][stage],'#72e8c0'))
    continuation=next((r for r in data['other_training'] if r['family']=='submission-grpo-continuation-v1' and r['arm']=='updates-96'),None)
    if continuation:
        rows.append(('Gemma visual · additional 96 GRPO',continuation['evaluations']['grpo'],'#b9f69a'))
    for row in sorted(data['frontier_perception'],key=lambda x:x['micro_f2']):
        rows.append((row['model'],row,'#e5bc82'))
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#eef5f4','axes.labelcolor':'#b0c4ca','xtick.color':'#b0c4ca','ytick.color':'#eef5f4','font.size':12})
    fig,(ax,errors)=plt.subplots(1,2,figsize=(16,9),facecolor='#10191d',gridspec_kw={'width_ratios':[1.5,1]})
    y=np.arange(len(rows));values=[100*r[1]['micro_f2'] for r in rows]
    for a in (ax,errors):
        a.set_facecolor('#10191d');a.invert_yaxis();a.tick_params(length=0,pad=10)
        for spine in a.spines.values():spine.set_visible(False)
    ax.barh(y,values,height=.53,color=[r[2] for r in rows]);ax.set_yticks(y,[r[0] for r in rows]);ax.set_xlim(0,100)
    ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);ax.set_xlabel('Detection micro-F2 (higher is better)',labelpad=14)
    ax.grid(axis='x',color='#30434a',alpha=.6);ax.set_axisbelow(True)
    for i,value in enumerate(values):ax.text(value+1.3,i,f'{value:.1f}%',va='center',fontsize=13,weight='bold')
    errors.barh(y-.14,[r[1]['fp'] for r in rows],height=.25,color='#d49182',label='False box claims')
    errors.barh(y+.14,[r[1]['fn'] for r in rows],height=.25,color='#71878d',label='Missed reference objects')
    errors.set_yticks(y,[]);errors.set_xlim(0,100);errors.set_xlabel('Counts (lower is better)',labelpad=14)
    for i,(_,r,_) in enumerate(rows):
        errors.text(r['fp']+1,i-.14,str(r['fp']),va='center',fontsize=10)
        errors.text(r['fn']+1,i+.14,str(r['fn']),va='center',fontsize=10)
    legend=errors.legend(loc='lower center',bbox_to_anchor=(.5,1.025),frameon=False,fontsize=11)
    for text in legend.get_texts():text.set_color('#eef5f4')
    fig.text(.045,.94,'AEROROZUM / MEASURED DEVELOPMENT RESULTS',color='#72e8c0',fontsize=13,weight='bold')
    student=max((r[1] for r in rows if r[0].startswith('Gemma')),key=lambda r:r['micro_f2'])
    frontier=max(data['frontier_perception'],key=lambda r:r['micro_f2'])
    fig.text(.045,.87,'Measured thermal perception after GRPO.',fontsize=29,weight='bold')
    fig.text(.045,.82,f'{100*student["micro_f2"]:.1f}% selected student F2 versus {100*frontier["micro_f2"]:.1f}% strongest frontier · development only',color='#b0c4ca',fontsize=14)
    fig.subplots_adjust(left=.29,right=.94,top=.74,bottom=.27,wspace=.18)
    fig.text(.045,.105,'Same 120 development images · 88 reference objects + 32 screened negatives · test scores excluded from selection',color='#b0c4ca',fontsize=11)
    fig.text(.045,.068,'Gemma: greedy 192-token output. Frontiers: native JSON, 8192 tokens including default reasoning. Not matched compute.',color='#b0c4ca',fontsize=11)
    fig.text(.045,.031,'Image-level perception only. Development selection across runs; no sealed-test or mission-policy superiority claim.',color='#b0c4ca',fontsize=11)
    fig.savefig(output/'development-comparison.png',dpi=120,facecolor=fig.get_facecolor())
    svg=output/'development-comparison.svg';fig.savefig(svg,facecolor=fig.get_facecolor(),metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (output/'development-comparison-provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sealed_test=False,matched_compute=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--latest',action='store_true');a=p.parse_args();(plot_latest if a.latest else plot)(a.source,a.output)
