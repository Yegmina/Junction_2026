#!/usr/bin/env python3
"""Render completed measurements; refuses partial validation snapshots."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--comparison',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 d=json.loads(Path(a.comparison).read_text())
 if d.get('status')!='completed' or d.get('validation_images')!=88:raise ValueError('Completed frozen 88-image comparison required')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=['Base Gemma 4','Real-flight SFT','SFT + GRPO'];keys=['base','real_sft','real_grpo'];rows=[d['versions'][k] for k in keys]
 if any(r['images']!=88 for r in rows):raise ValueError('Incomplete model comparison')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'#10191a','axes.facecolor':'#10191a','text.color':'#e0e9df','axes.labelcolor':'#a7b9b3','xtick.color':'#cbd9cd','ytick.color':'#8ea39a','axes.edgecolor':'#35453d','savefig.facecolor':'#10191a'})
 fig=plt.figure(figsize=(14,8),dpi=160);gs=fig.add_gridspec(2,2,height_ratios=[1.5,1],hspace=.55,wspace=.28)
 for ax,metric,title in [(fig.add_subplot(gs[0,0]),'micro_f2','Detection F2 · IoU ≥ 0.5'),(fig.add_subplot(gs[0,1]),'recall','Annotation recall · IoU ≥ 0.5')]:
  vals=[100*r[metric] for r in rows];ax.bar(names,vals,color=['#8aa69a','#c8e6a2','#d3b489'],width=.5)
  for i,value in enumerate(vals):ax.text(i,value+3,f'{value:.2f}%',ha='center',color='#e7eddf',fontsize=14)
  ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title(title,loc='left',pad=15,color='#dae9d3');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#27362e',alpha=.7);ax.set_axisbelow(True)
 ax=fig.add_subplot(gs[1,:]);ax.axis('off');headers=['Model','Matches / 88','Extra boxes','Empty responses','Invalid JSON','Matched IoU'];cells=[]
 for name,r in zip(names,rows):cells.append([name,str(r['tp']),str(r['fp']),str(r['empty_predictions']),str(r['invalid_json']),f"{r['mean_matched_iou']:.3f}" if r['tp'] else 'No matches'])
 table=ax.table(cellText=cells,colLabels=headers,loc='center',cellLoc='center',colLoc='center',bbox=[0,.05,1,.9]);table.auto_set_font_size(False);table.set_fontsize(11)
 for (i,j),cell in table.get_celld().items():cell.set_facecolor('#1a2922' if i==0 else '#13201a');cell.set_edgecolor('#34443a');cell.get_text().set_color('#cadbc9')
 fig.suptitle('Held-out zone 7: measured cross-domain annotation agreement',x=.07,ha='left',fontsize=19,color='#e6f0db');fig.text(.07,.91,'88 positive images · 30 original manual / 58 reviewed AI · one frozen prompt and greedy decoding',fontsize=11,color='#8da89a')
 fig.text(.07,.035,'Positive images only: negative-image false-alarm rate is unavailable. Test zones 8–9 remain untouched.\nAgreement with supplied boxes does not establish independently verified physical mine accuracy.',fontsize=10,color='#b6ac98',linespacing=1.7)
 fig.subplots_adjust(left=.07,right=.96,bottom=.16,top=.84);Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output);print(a.output)
if __name__=='__main__':main()
