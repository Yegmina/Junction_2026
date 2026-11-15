"""Export every terminal native outcome and an explicitly incomplete score grid."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path('/opt/aerorozum');NIGHT=ROOT/'artifacts/overnight-20261115T000942'
MODELS=['oracle','gemini-4-carbon','gemini-4-argon','gemma-local-deployed','gemma-native16k']
LABELS=['Vision reference','Carbon · 131k / 32k','Argon · 131k / 32k','Gemma JSON · 24k / 4k','Gemma native · 16k / 4k']
WORLDS=[(seed,phase) for seed in range(940201,940211) for phase in ['dawn','warming']]


def main():
    from aerorozum.common import save,utcnow
    out=NIGHT/'native-outcomes-v1';out.mkdir(exist_ok=True)
    records=[]
    for path in (ROOT/'harbor/jobs').glob('overnight-*/result.json'):
        result=json.loads(path.read_text());stats=result.get('stats',{})
        if not result.get('finished_at') or stats.get('n_running_trials') or stats.get('n_pending_trials'):continue
        model,phase,seed=path.parent.name.removeprefix('overnight-').rsplit('-',2);seed=int(seed)
        if model not in MODELS or (seed,phase) not in WORLDS:continue
        trials=list(path.parent.glob('*/verifier/details.json'))
        details=json.loads(trials[0].read_text()) if len(trials)==1 else {}
        scores=[m['mean'] for e in stats.get('evals',{}).values() for m in e.get('metrics',[]) if isinstance(m.get('mean'),(int,float))]
        error=stats.get('n_errored_trials',0)>0
        over=details.get('flight_elapsed_s',0)>details.get('budget_s',75)+1e-6
        reason='infrastructure_error' if error else 'flight_budget_exceeded' if over else 'other_completion_gate' if details.get('completed') is False else 'completed_task'
        records.append(dict(model=model,phase=phase,seed=seed,reward=scores[0] if len(scores)==1 else None,
            infrastructure_error=error,budget_exceeded=over,outcome_category=reason,verifier=details,
            terminal_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),finished_at=result['finished_at']))
    receipt=dict(at=utcnow(),planned_worlds=[dict(seed=s,phase=p) for s,p in WORLDS],models=MODELS,records=records,
        complete_by_model={m:sum(r['model']==m for r in records)==len(WORLDS) for m in MODELS},
        outcome_counts={m:dict(Counter(r['outcome_category'] for r in records if r['model']==m)) for m in MODELS},
        context_output_tokens={'oracle':None,'gemini-4-carbon':[131072,32768],'gemini-4-argon':[131072,32768],
                              'gemma-local-deployed':[24576,4096],'gemma-native16k':[16384,4096]},
        no_outcome_based_retries=True,full_subset_means_not_reported=True,
        interpretation='Native Harbor rewards for inserted synthetic objects in experimental composed imagery. Every terminal outcome is shown; errors are distinct from clean zero scores. Context and output budgets differ. Native Gemma also changes the tool interface; no isolated model-training effect is claimed.')
    save(out/'summary.json',receipt)
    lookup={(r['model'],r['seed'],r['phase']):r for r in records};values=np.full((len(MODELS),len(WORLDS)),np.nan)
    for i,m in enumerate(MODELS):
        for j,(seed,phase) in enumerate(WORLDS):
            r=lookup.get((m,seed,phase))
            if r and isinstance(r['reward'],(int,float)):values[i,j]=r['reward']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(16,4.8),layout='constrained');cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#e5e7eb')
    image=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    for i,m in enumerate(MODELS):
        for j,(seed,phase) in enumerate(WORLDS):
            r=lookup.get((m,seed,phase));label='—' if not r else 'E' if r['infrastructure_error'] else f"{r['reward']:.2f}"+('B' if r['budget_exceeded'] else '') if r['reward'] is not None else '?'
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if r and not r['infrastructure_error'] and (r['reward'] or 0)>.55 else '#ad3d14' if r and r['infrastructure_error'] else '#182b35')
    ax.set_yticks(range(len(MODELS)),[LABELS[i]+' · '+str(sum(r['model']==m for r in records))+'/20 terminal' for i,m in enumerate(MODELS)])
    ax.set_xticks(range(len(WORLDS)),[str(seed)[-3:]+' '+('D' if p=='dawn' else 'W') for seed,p in WORLDS],rotation=50,ha='right')
    ax.set_xticks(np.arange(-.5,len(WORLDS),1),minor=True);ax.set_yticks(np.arange(-.5,len(MODELS),1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',bottom=False,left=False)
    fig.colorbar(image,ax=ax,label='Native mission reward',shrink=.8)
    fig.suptitle('Investigation rewards · every terminal outcome retained',fontsize=18,fontweight='bold',x=.02,ha='left')
    fig.supxlabel('Snapshot '+receipt['at'][:19]+' UTC · D = dawn, W = warming · E = infrastructure error · B = flight budget exceeded · — = no terminal result yet\nFixed 75-second simulated flight budget. Model budgets differ. Missing cases are never scored as zero.\nExperimental composed thermal imagery; this is an agent investigation benchmark, not field detection accuracy.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig(out/('native-outcomes-v1.'+suffix),dpi=160)
    print(json.dumps({'at':receipt['at'],'terminal':len(records),'outcome_counts':receipt['outcome_counts']}))


if __name__=='__main__':main()
