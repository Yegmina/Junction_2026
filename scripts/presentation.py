"""Build a factual, presentation-ready progress deck from recorded artifacts."""
from __future__ import annotations
import html
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from aerorozum.common import ROOT,data_root,save,utcnow

OUT=ROOT/'docs/presentation';ASSETS=OUT/'assets'
OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)
root=data_root();state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
audit=json.loads((root/'audits/dataset.json').read_text()) if (root/'audits/dataset.json').exists() else {}
jobs=[json.loads(p.read_text()) for p in sorted((root/'jobs').glob('*.json'))]
response_dirs=[root/'gemini/responses',*(root/'labelling').glob('*/labels/provider/responses'),*(root/'backgrounds').glob('*/labels/provider/responses')]
response_records=[json.loads(p.read_text()) for folder in response_dirs for p in folder.glob('*.json')]
responses=list({(r.get('id'),r.get('created_at'),r.get('model')):r for r in response_records}.values())
plt.rcParams.update({'figure.facecolor':'#101a1e','axes.facecolor':'#101a1e','text.color':'#e1ebe4','axes.labelcolor':'#b1c6b8','xtick.color':'#b1c6b8','ytick.color':'#b1c6b8','axes.edgecolor':'#53645a','font.size':12})

fig,axes=plt.subplots(1,2,figsize=(12,4));fig.suptitle('Real labelled dataset · audited, not independently verified identities')
axes[0].bar(['Manual','Reviewed AI'],[audit.get('provenance',{}).get('original_manual',0),audit.get('provenance',{}).get('reviewed_ai',0)],color=['#c4e997','#7ea99b']);axes[0].set_ylabel('Images')
axes[1].bar(['Train zones 1–5','Validation zone 7','Test zones 8–9'],[audit.get('counts',{}).get(k,0) for k in ['train_images','validation_images','test_images']],color=['#c4e997','#e8bc84','#7ea99b']);axes[1].tick_params(axis='x',rotation=10)
fig.tight_layout();fig.savefig(ASSETS/'03-dataset-audit.png',dpi=150);plt.close(fig)

physics_image=None
fields=sorted((root/'physics').glob('**/field.npz'))+sorted((root/'physics').glob('*.npz'))
for path in fields:
    try:
        with np.load(path,allow_pickle=False) as d:
            if len(d['times_s'])<3:continue
            t=d['times_s']/3600;temp=d['temperature_k'];moist=d['moisture']
            if t[-1]<10:continue
            fig,axes=plt.subplots(1,3,figsize=(15,4));fig.suptitle('Actual coupled simulation output · numerical acceptance pending')
            im=axes[0].imshow(temp[-1]-273.15,cmap='inferno',origin='lower');fig.colorbar(im,ax=axes[0],label='Surface °C');axes[0].set_title(f'Final surface · {t[-1]:.1f} virtual hours')
            axes[1].plot(t,temp.mean(axis=(1,2))-273.15,color='#e8bc84');axes[1].set(xlabel='Virtual hours',ylabel='Mean surface °C')
            axes[2].plot(t,moist.mean(axis=(1,2)),color='#90c6cf');axes[2].set(xlabel='Virtual hours',ylabel='Mean exported moisture variable')
            fig.tight_layout();physics_image='04-coupled-physics-output.png';fig.savefig(ASSETS/physics_image,dpi=150);plt.close(fig)
            break
    except (KeyError,ValueError,OSError):continue

real=Path('/home/yehormsi/Downloads/gemini_parallel_flight_test/evidence/frame_002695.jpg')
if real.exists():shutil.copyfile(real,ASSETS/'05-real-flight-reference.jpg')
if real.exists() and (ASSETS/'02-thermal-appearance-randomized.png').exists():
    fig,axes=plt.subplots(1,2,figsize=(12,5));fig.suptitle('Appearance comparison · synthetic texture is not validated thermal physics')
    axes[0].imshow(Image.open(real),cmap='gray',vmin=0,vmax=255);axes[0].set_title('Original unlabelled drone frame')
    axes[1].imshow(Image.open(ASSETS/'02-thermal-appearance-randomized.png'),cmap='gray',vmin=0,vmax=255);axes[1].set_title('Random-phase demo fitted to 32 real frames')
    for a in axes:a.axis('off')
    fig.tight_layout();fig.savefig(ASSETS/'06-real-synthetic-appearance.png',dpi=150);plt.close(fig)

timestamp=utcnow();summary={'recorded_at':timestamp,'real_images':audit.get('counts',{}).get('images',0),'boxes':audit.get('counts',{}).get('boxes',0),'gemini_completed_requests':len(responses),'gemini_tokens':sum(r.get('usage',{}).get('total_token_count') or 0 for r in responses),'jobs':[{'kind':j.get('kind'),'status':j.get('status'),'completed':j.get('completed'),'total':j.get('total')} for j in jobs],'research_accepted':state.get('physics',{}).get('research_accepted',False),'full_training_completed':False}
save(OUT/'snapshot.json',summary)
slides=[
 ('AeroRozum · A world to learn from','Physics-based thermal survey simulation, high-volume Gemini supervision, and Gemma learning.','Build agents that gather evidence and handle uncertainty, then measure them on unseen scenes.',None),
 ('The data we actually have',f"{summary['real_images']} images · {summary['boxes']} annotated boxes",'All annotated images contain positives. Provenance includes manual and reviewed AI labels. Whole-zone holdouts reduce background leakage.','03-dataset-audit.png'),
 ('The system','Physics → camera → Gemini teacher → Gemma SFT → Gemma RL → frozen evaluation','Physics owns synthetic truth. Gemini supplies visible-evidence descriptions and public-tool demonstrations. Gemma is the trainable model.','architecture.svg' if (ASSETS/'architecture.svg').exists() else None),
 ('Coupled thermal physics','Sun, sky radiation, wind, heat transfer, moisture redistribution and evaporation','Pinned DuMux/DUNE upstream evaporation regression passed. Custom 3D fields and refinement checks are recorded separately. Field validation has not been performed.',physics_image),
 ('Visible campaign environment','Six ten-minute sorties across one virtual day','The agent chooses when to fly, where to inspect, when to revisit and what evidence to submit. Movement, turnaround and camera observations have explicit costs.','07-dashboard-current.png' if (ASSETS/'07-dashboard-current.png').exists() else '01-dashboard-initial-demo.png'),
 ('An appearance mistake we caught','From procedural stripes to actual recorded thermal pixels','A statistical texture still looked synthetic. The live demo now samples a coherent quilt of original thermal frames, preserving their grayscale values. Recorded backgrounds remain unvalidated composites.','10-recorded-ground-comparison.png' if (ASSETS/'10-recorded-ground-comparison.png').exists() else '06-real-synthetic-appearance.png'),
 ('Gemini is already working',f"{summary['gemini_completed_requests']} saved responses · {summary['gemini_tokens']:,} recorded tokens",'The separate hackathon API key powers labels, reviews and campaign agents. These cached-success totals exclude raw failed attempts, which are retained separately. Teacher outputs do not replace physical truth.',None),
 ('The research now streams','Arriving evidence → frozen batch → SFT → GRPO → development gate','The first accepted pair can release auxiliary sensor crops without an all-184 barrier. Real labels and reviewed anchors already feed parallel GPU passes. Numerical acceptance, data splits and promotion checks remain mandatory.','20-pipeline-overview.png' if (ASSETS/'20-pipeline-overview.png').exists() else '18-continuous-pipeline-desktop.png'),
 ('Actual inputs and progress','Measured resources, physical fields and recorded evidence','The dashboard separates experimental physical maps, synthetic crops and recorded training evidence. The PCR7 integrity alert stays open; its cause is unconfirmed and its baseline was not reset.','21-training-input-evidence.png' if (ASSETS/'21-training-input-evidence.png').exists() else None),
 ('Gemma is the model we train','Gemma 4 12B · base versus SFT versus GRPO',f"Real-transfer run: {state.get('training',{}).get('sft_updates',0)} SFT updates and {state.get('training',{}).get('grpo_updates',0)} GRPO updates recorded. The machine-labelled pilot is separate from the full three-seed physics study. Training reward alone does not establish better detection.",'12-gemma-real-transfer-training.png' if (ASSETS/'12-gemma-real-transfer-training.png').exists() else None),
 ('The first held-out result','The native-flight transfer pilot did not generalize to the reviewed ZIP domain','On 88 zone-7 positive images: base F2 0.0557; SFT 0; GRPO 0. Both adapters returned empty lists throughout. This is annotation agreement on a positive-only set, not independently verified mine accuracy. The negative result is retained in full.','14-validation-comparison.png' if (ASSETS/'14-validation-comparison.png').exists() else None),
 ('Robotics ER 2 labelling','Native thermal footage sampled at five frames per second','Use the exact Robotics ER 2 beta endpoint. Confidence ≥0.6 is accepted under the user-defined training policy, with machine-label provenance. Save source hashes, labels and response logs back to the source bucket.',None),
 ('The experiment is frozen','30,000 images · 512 training campaigns · 64 validation campaigns','72 fast-test plus 24 direct-test campaigns. Matched original, relocated and target-free layouts. These are protocol targets, not completed result counts.',None),
 ('Reward and integrity','Unique-object F2 + qualified coverage','One-to-one evidence matching penalizes duplicate claims. Coverage spans morning, heating and cooling. No-op, box spam, hidden truth access and repeat-hover exploits are tested.',None),
 ('Engineering evidence','Reproducible tests, model calls, solver logs and bounded cloud jobs','Four A100 40 GB GPUs were provisioned for a bounded pilot. Actual progress appears in the dashboard. Each partner submission requires demonstrated use, not a badge.',None),
 ('What the results must establish','Does training improve evidence-backed detection and survey decisions?','Report held-out performance, false alarms, relocation robustness, direct-versus-fast fidelity, resource costs, failed checks and remaining uncertainty.',None),
]
md=['# AeroRozum presentation source',f'\nSnapshot: {timestamp}\n','Actual completed results and planned protocol targets are explicitly distinguished.\n']
sections=[]
for i,(title,lead,body,img) in enumerate(slides,1):
    md.extend([f'## {i}. {title}',lead,'',body,''])
    if img:md.extend([f'![{title}](assets/{img})',''])
    figure=f'<img src="assets/{html.escape(img)}" alt="{html.escape(title)}">' if img else ''
    sections.append(f'<section><small>AEROROZUM · RESEARCH JOURNAL / {i:02d}</small><h1>{html.escape(title)}</h1><h2>{html.escape(lead)}</h2><p>{html.escape(body)}</p>{figure}<footer>{html.escape(timestamp)} · Evidence over assumptions</footer></section>')
(OUT/'slides.md').write_text('\n'.join(md))
(OUT/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AeroRozum · Research presentation</title><style>body{margin:0;background:#0d1518;color:#e7eee9;font:20px system-ui}section{box-sizing:border-box;min-height:100vh;padding:5vh 7vw;border-bottom:1px solid #35443d;position:relative}small,footer{color:#a5bfaa;font-size:13px;letter-spacing:.14em}h1{font-size:clamp(30px,4vw,62px);max-width:1100px;margin:.6em 0}h2{font-size:clamp(23px,2.4vw,36px);font-weight:400;color:#c8e79e}p{max-width:1000px;line-height:1.6;color:#b1c2b8}img{display:block;max-width:100%;max-height:57vh;object-fit:contain;margin:2em auto}footer{margin-top:3em}@media print{section{break-after:page;min-height:95vh}}</style>'+''.join(sections)+'<script>document.addEventListener("keydown",e=>{const a=[...document.querySelectorAll("section")],i=a.findIndex(s=>s.getBoundingClientRect().bottom>100);if(["ArrowRight","PageDown"," "].includes(e.key)){e.preventDefault();a[Math.min(a.length-1,i+1)].scrollIntoView()}if(["ArrowLeft","PageUp"].includes(e.key)){e.preventDefault();a[Math.max(0,i-1)].scrollIntoView()}})</script></html>')
print(json.dumps(summary,indent=2))
