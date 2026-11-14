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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(a.source,a.output)
