"""Publication-style plots of measured paired development generations."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    output=Path('docs/evidence');data=json.loads((output/'robotics-thinking-study-v1.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    levels=['minimal','medium','high'];colors=['#087e8b','#c26637'];prompts=['thermal-v2-recall-calibrated','thermal-v3-two-scale-review']
    for index,prompt in enumerate(prompts):
        groups=[[a for a in data['arms'] if a['prompt_version']==prompt and a['thinking_level']==level] for level in levels]
        for ax,metric,title in [(axes[0,0],lambda a:a['all_development']['micro_f2'],'F2 · all 120 development references'),
                                (axes[0,1],lambda a:a['original_manual_only']['micro_f2'],'F2 · 30 original-manual references only'),
                                (axes[1,0],lambda a:a['mean_rpc_elapsed_s'],'Mean API response time, seconds'),
                                (axes[1,1],lambda a:a['thought_tokens']/a['images'],'Thought-token usage per image')]:
            values=[[metric(a) for a in group] for group in groups];means=[statistics.mean(v) for v in values];std=[statistics.stdev(v) for v in values]
            x=np.arange(3)+(index-.5)*.06
            ax.errorbar(x,means,yerr=std,marker='o',capsize=4,color=colors[index],lw=2,label='V2 · recall calibrated' if index==0 else 'V3 · two-scale review')
            for k,replicas in enumerate(values):ax.scatter(np.full(len(replicas),x[k]),replicas,color=colors[index],s=12,alpha=.5)
            ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.set_xticks(range(3),[x.upper() for x in levels]);ax.grid(axis='y',alpha=.15);ax.set_xlim(-.35,2.35)
    for ax in axes[0]:ax.set_ylim(0,1)
    axes[0,0].legend(frameon=False,loc='lower right',fontsize=9)
    fig.suptitle('More thinking helps here. The prompt still matters.',fontsize=19,fontweight='bold',x=.02,ha='left')
    fig.supxlabel('Three generation repeats per configuration · same 65,536-token cap · confidence ≥ 0.60 · IoU ≥ 0.50\nError bars: across-repeat standard deviation. Development references include 30 original-manual, 58 reviewed-AI and 32 machine-screened-zero images.\nResponse time excludes quota waiting. These are label-agreement results, not confirmed physical mine accuracy.',fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(output/('robotics-thinking-study-v1.'+suffix),dpi=180)
    svg=output/'robotics-thinking-study-v1.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    summary=[]
    for prompt in prompts:
        for level in levels:
            arms=[a for a in data['arms'] if a['prompt_version']==prompt and a['thinking_level']==level]
            summary.append(dict(prompt=prompt,thinking_level=level,mean_development_f2=statistics.mean(a['all_development']['micro_f2'] for a in arms),
                std_development_f2=statistics.stdev(a['all_development']['micro_f2'] for a in arms),
                mean_original_manual_f2=statistics.mean(a['original_manual_only']['micro_f2'] for a in arms),
                mean_rpc_s=statistics.mean(a['mean_rpc_elapsed_s'] for a in arms)))
    (output/'robotics-thinking-study-v1-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
