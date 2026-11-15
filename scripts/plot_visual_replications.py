"""Plot the recorded three-seed development comparison without selecting seeds."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/evidence');data=json.loads((out/'visual-replications-summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
colors=['#087e8b','#a64b35','#654b98']
for row,color in zip(data['seeds'],colors):
 values=row['evaluations'];axes[0].plot(range(3),[v['micro_f2'] for v in values],marker='o',lw=2,color=color,label='Seed '+str(row['seed'])+' · final '+f"{values[-1]['micro_f2']:.3f}")
 axes[1].plot(range(3),[100*v['negative_false_alarm_rate'] for v in values],marker='o',lw=2,color=color)
axes[0].set_title('Development detection agreement · F2',loc='left',fontweight='bold');axes[0].set_ylim(0,1);axes[0].legend(frameon=False)
axes[1].set_title('Candidate-bearing model-zero reference frames',loc='left',fontweight='bold');axes[1].set_ylabel('% of 32 model-zero references');axes[1].set_ylim(0,25)
for ax in axes:
 ax.set_xticks(range(3),['480 SFT','+48 GRPO','+144 GRPO total']);ax.grid(axis='y',alpha=.2)
means=[statistics.mean(r['evaluations'][i]['micro_f2'] for r in data['seeds']) for i in range(3)]
fig.suptitle('Three seeds improve F2; false alarms also change.',fontsize=18,fontweight='bold',x=.02,ha='left')
fig.supxlabel('602 training images · same 120 development images (30 original-manual, 58 reviewed-AI, 32 model-zero references)\nAll three predeclared seeds shown. Repeated development evaluation does not establish field generalization.\nSeed-17 additional 96-SFT control: F2 '+f"{data['additional_96_sft_seed17']['micro_f2']:.3f}"+'. Controls match update counts, not compute. The deployed model is unchanged.',fontsize=9)
for suffix in ['png','pdf','svg']:fig.savefig(out/('visual-replications-v1.'+suffix),dpi=180)
p=out/'visual-replications-v1.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
print(json.dumps({'mean_development_f2':means,'all_predeclared_seeds_shown':True,'production_model_changed':False}))
