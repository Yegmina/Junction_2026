"""Build report tables and a source-hashed experiment ledger from recorded runs."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/report'
E = ROOT / 'docs/evidence'
sources = {}
ledger = []


def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def esc(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def table(name, headings, rows, alignment):
    text = '\\begin{tabular}{@{}' + alignment + '@{}}\n\\toprule\n'
    text += ' & '.join(headings) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (OUT / 'tables' / (name + '.tex')).write_text(text)


native = read(OUT / 'data/native-outcomes-summary.json')
refs = {(r['seed'], r['phase']): r['reward'] for r in native['records'] if r['model'] == 'oracle' and not r['infrastructure_error']}
names = dict(oracle='Public-image ref.', **{'gemini-4-carbon': 'Carbon', 'gemini-4-argon': 'Argon',
             'gemma-local-deployed': 'Gemma legacy', 'gemma-native16k': 'Gemma native16k'})
rows = []
for model in native['models']:
    rr = [r for r in native['records'] if r['model'] == model]
    clean = [r for r in rr if not r['infrastructure_error']]
    paired = [r for r in clean if (r['seed'], r['phase']) in refs]
    rows.append([names[model], len(rr), len(rr)-len(clean), len(paired),
                 f"{st.mean(r['reward'] for r in paired):.3f}" if paired else '--',
                 f"{st.mean(refs[(r['seed'],r['phase'])] for r in paired):.3f}" if paired else '--'])
    for r in rr:
        ledger.append(dict(family='native-75s-multiseed', variant=model, seed=r['seed'], phase=r['phase'],
                           status=r['outcome_category'], metric='mission_reward', value=r['reward'] if not r['infrastructure_error'] else '',
                           source='docs/report/data/native-outcomes-summary.json', source_sha256=r['terminal_result_sha256']))
table('native', ['Agent', 'Term.', 'Err.', 'Pairs', 'Agent', 'Ref.'], rows, 'lrrrrr')

measure = read(E / 'submission-measurements-latest.json')
for r in measure['trials']:
    ledger.append(dict(family='earlier-native', variant=r['job'], status=r['exception_type'] or 'clean_terminal',
                       metric='mission_reward', value=(r.get('rewards') or {}).get('reward', ''),
                       source='docs/evidence/submission-measurements-latest.json', source_sha256=r['result_sha256']))
byjob = {r['job']: r for r in measure['trials']}
rows = []
for job,label in [('reference-budgeted-s730171','Reference A'),('reference-budgeted-s730172','Reference B'),
                  ('argon-budgeted-32k-s730171','Argon 32k A'),('argon-budgeted-32k-s730172','Argon 32k B'),
                  ('carbon-budgeted-32k-s730171','Carbon 32k A'),('carbon-budgeted-32k-s730172','Carbon 32k B$^*$')]:
    r=byjob[job]; rows.append([label, f"{r['details']['flight_elapsed_s']:.3f}", f"{r['rewards']['reward']:.3f}"])
table('matched', ['Run', 'Flight seconds', 'Reward'], rows, 'lrr')

rep = read(E / 'visual-replications-summary.json')
rows = []
for r in rep['seeds']:
    rows.append([r['seed']] + [f"{v['micro_f2']:.3f}" for v in r['evaluations']])
    for v in r['evaluations']:
        ledger.append(dict(family='602-image-replication', variant=v['stage'], seed=r['seed'], status='completed',
                           metric='development_F2', value=v['micro_f2'], source='docs/evidence/visual-replications-summary.json'))
rows.append(['Mean']+[f"{st.mean(r['evaluations'][i]['micro_f2'] for r in rep['seeds']):.3f}" for i in range(3)])
table('replications', ['Seed','SFT','+48 GRPO','+144 total'], rows, 'lrrr')

expanded = read(E / 'expanded-comparison-v1.json')
rows = []
for r in expanded['seeds']:
    assert all(s['state']=='completed' for s in r['stages'])
    rows.append([r['seed']] + [f"{s['micro_f2']:.3f}" for s in r['stages']])
    for s in r['stages']:
        ledger.append(dict(family='2400-image-comparison', variant=s['stage'], seed=r['seed'], status=s['state'],
                           metric='development_F2', value=s['micro_f2'], actual_updates=s['actual_updates'],
                           requested_updates=s['maximum_requested_updates'], source=s['source'], source_sha256=expanded['source_sha256'][s['source']]))
rows.append(['Mean']+[f"{st.mean(r['stages'][i]['micro_f2'] for r in expanded['seeds']):.3f}" for i in range(4)])
table('expanded', ['Seed','SFT','GRPO','Extra SFT','Extra GRPO'], rows, 'lrrrr')

rows=[]
held={r['model']:r for r in measure['held_out_perception']}
for model,label in [('gemma-base','Gemma base'),('gemma-parent','Gemma parent'),('gemma','Gemma +96 GRPO'),
                    ('gemini-robotics-er-2-preview','Robotics ER 2'),('gemini-4-carbon','Carbon'),('gemini-4-argon','Argon')]:
    r=held[model];assert r['examples']==182
    rows.append([label,f"{r['micro_f2']:.3f}",r['tp'],r['fp'],r['fn']])
    ledger.append(dict(family='frozen-182-image-heldout',variant=model,status='completed',metric='heldout_F2',value=r['micro_f2'],
                       source='docs/evidence/submission-measurements-latest.json',source_sha256=r['source_sha256']))
table('heldout',['Model','F$_2$','TP','FP','FN'],rows,'lrrrr')

early=read(E/'measured-refinements.json')
early_rows=[]
for r in early['runs']:
    label=r['family'].replace('submission-refinement-','')+'/'+r['arm']
    vals=r['evaluations'];early_rows.append([esc(label),f"{vals['sft_evaluation']['micro_f2']:.3f}",f"{vals['grpo_evaluation']['micro_f2']:.3f}"])
    for phase,v in vals.items():
        tr=r.get('training',{}).get(phase.removesuffix('_evaluation'),{})
        ledger.append(dict(family=r['family'],variant=r['arm']+'/'+phase,status='completed',metric='development_F2',value=v['micro_f2'],
                           actual_updates=tr.get('optimizer_updates',''),requested_updates=tr.get('requested_optimizer_updates',''),
                           source='docs/evidence/measured-refinements.json',source_sha256=r['source_result_sha256']))
short={'submission-reward-curriculum-v1':'Dense reward 16/64','submission-visual-projection-v2':'Visual',
       'submission-refinement-v3':'Language continuation','submission-visual-continuation-v1':'Visual continuation',
       'submission-grpo-continuation-v1':'GRPO continuation'}
for r in measure['other_training']:
    vals=r['evaluations'];sft=vals.get('sft') or vals.get('base_control');grpo=vals.get('grpo')
    early_label=short.get(r['family'],r['family'])
    if r['arm'] != r['family']: early_label += '/'+r['arm']
    early_rows.append([esc(early_label), f"{sft['micro_f2']:.3f}" if sft else '--', f"{grpo['micro_f2']:.3f}" if grpo else '--'])
    for phase,v in vals.items():
        tr=r.get('training',{}).get(phase,{})
        ledger.append(dict(family=r['family'],variant=r['arm']+'/'+phase,status='completed',metric='development_F2',value=v['micro_f2'],
                           actual_updates=tr.get('optimizer_updates',''),requested_updates=tr.get('requested_optimizer_updates',''),
                           source='docs/evidence/submission-measurements-latest.json',source_sha256=r['source_sha256']))
table('early',['Recorded arm','SFT/base','GRPO'],early_rows,'lrr')

thinking=read(E/'robotics-thinking-study-v1.json');summary=read(E/'robotics-thinking-study-v1-summary.json')
table('thinking',['Prompt','Thinking','Mean F$_2$','SD','RPC s'],[[r['prompt'].split('-')[1].upper(),r['thinking_level'],f"{r['mean_development_f2']:.3f}",f"{r['std_development_f2']:.3f}",f"{r['mean_rpc_s']:.2f}"] for r in summary],'llrrr')
for r in thinking['arms']:
    ledger.append(dict(family='Gemini-prompt-thinking',variant=r['prompt_version']+'/'+r['thinking_level'],seed=r['replica'],
                       status='completed',metric='development_F2',value=r['all_development']['micro_f2'],source='docs/evidence/robotics-thinking-study-v1.json'))

physics=read(OUT/'data/physics-inventory.json')
for r in physics['cases']:
    d=r['record'];ledger.append(dict(family='physics-case-state',variant=r['path'],status=d.get('status','unknown'),
                                   metric='elapsed_wall_s',value=d.get('elapsed_wall_s',''),source='docs/report/data/physics-inventory.json',source_sha256=r['source_sha256']))
wrap=read(E/'wrap-up-training-v1.json')
for seed,r in wrap['results'].items():
    ledger.append(dict(family='4688-image-HIGH-teacher',variant='SFT',seed=seed,status=r['status'],actual_updates=r['saved_optimizer_updates'],
                       requested_updates=r['planned_optimizer_updates'],metric='not_evaluated',value='',source='docs/evidence/wrap-up-training-v1.json'))
for name,variant in [('paired-class-replay-protocol.json','balanced-vs-positive-replay'),('snapshot-monitor-qualification.json','fixed-snapshot-evaluation')]:
    read(E/name);ledger.append(dict(family='unexecuted-followup',variant=variant,status='not_run_after_user_wrap_up',metric='not_evaluated',value='',source='docs/evidence/'+name))
for name in ['temporal-review-diagnostic-v1.json','background-review-final-v1.json','high-thinking-primary-first-pass.json',
             'high-thinking-requalified-first-pass.json','high-thinking-bounded-recovery.json','physics-full-day-diagnostic-receipt.json',
             'gradient-monitor-benchmark.json','association-audit.json','opencode-wizard-terminal-fix-v3.json','visual-matched-update-control.json']:
    read(E/name)
fields=['family','variant','seed','phase','status','metric','value','actual_updates','requested_updates','source','source_sha256']
with (OUT/'experiment-ledger.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(ledger)
manifest=dict(scope='Hackathon artifact snapshot; rows may share parents and are not independent trials. Physics rows are recorded case states, not numerical acceptance certificates.',
              source_sha256=sources,ledger_rows=len(ledger),native_multiseed_trials=len(native['records']),physics_case_state_records=len(physics['cases']),
              training_and_mission_metrics_separate=True,field_accuracy_claimed=False)
(OUT/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'}))
