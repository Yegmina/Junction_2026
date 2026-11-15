"""Build a dated presentation from actual local evidence, without new inference."""
from __future__ import annotations
import bisect, collections, html, json, shutil, subprocess, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from aerorozum.common import ROOT, data_root

OUT = ROOT/'docs/presentation'
ASSETS = OUT/'assets'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(size, bold=False): return ImageFont.truetype(BOLD if bold else FONT, size)
def esc(value): return html.escape(str(value))

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    root = data_root()
    with urllib.request.urlopen('http://127.0.0.1:18901/api/pipeline', timeout=20) as response:
        state = json.load(response)
    stamp = datetime.now(timezone.utc).isoformat()
    (OUT/'live-briefing-snapshot.json').write_text(json.dumps(state, indent=2))
    index = json.loads((root/'pipeline/inputs/real.json').read_text())
    groups = collections.defaultdict(list)
    for row in index['images']:
        if row.get('split') == 'train' and row.get('provenance') == 'user_accepted_model' and row.get('source', {}).get('object') and row.get('source_offset_s') is not None:
            groups[row['source']['object']].append(row)
    source, rows = max(groups.items(), key=lambda item: sum(bool(r.get('boxes')) for r in item[1]))
    rows.sort(key=lambda r: r['source_offset_s'])
    offsets = [r['source_offset_s'] for r in rows]
    # Select a real 20-second window with many saved detections; never combine sources.
    start = max((round(o, 1) for o in offsets if o+20 <= offsets[-1]),
                key=lambda o: sum(bool(r.get('boxes')) for r in rows if o <= r['source_offset_s'] < o+20))
    clip_rows = [r for r in rows if start <= r['source_offset_s'] < start+20]
    chosen = max(clip_rows, key=lambda r: max((b.get('confidence', 0) for b in r.get('boxes', [])), default=0))
    def annotated(row):
        image = Image.open(root/row['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        for box in row.get('boxes', []):
            b = box['bbox']; xy = (b['x_min']*image.width/1000, b['y_min']*image.height/1000, b['x_max']*image.width/1000, b['y_max']*image.height/1000)
            draw.rectangle(xy, outline='#b8f394', width=2)
            label = f"candidate {box.get('confidence', 0):.2f}"
            x, y = max(0, xy[0]), max(0, xy[1]-19)
            draw.rectangle((x,y,min(image.width,x+130),y+18),fill='#071b17')
            draw.text((x+3,y+1), label, font=font(12,True),fill='#b8f394')
        return image
    annotated(chosen).save(ASSETS/'25-thermal-candidate.png')
    Image.open(root/chosen['image']).save(ASSETS/'25-thermal-original.png')
    physical = [m for m in state.get('media', []) if m.get('kind')=='physics' and 'v2' in m.get('case_id','')]
    physical = next((m for m in physical if m.get('case_id')=='axisym-v2-present'), physical[0] if physical else next(m for m in state['media'] if m.get('kind')=='physics'))
    shutil.copyfile(root/physical['image'], ASSETS/'26-actual-physics.png')
    video = ASSETS/'thermal-evidence-20s.mp4'
    working = video.with_name('thermal-evidence-20s.rendering.mp4')
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','5','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-vf','fps=25','-pix_fmt','yuv420p','-movflags','+faststart',str(working)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    used = []
    for tick in range(100):
        offset = start+tick/5
        nearest = min(rows, key=lambda r: abs(r['source_offset_s']-offset))
        available = abs(nearest['source_offset_s']-offset) <= .14
        image = Image.new('RGB',(1920,1080),'#091918'); draw = ImageDraw.Draw(image)
        draw.text((70,48),'AEROROZUM',font=font(25,True),fill='#bcf29e')
        draw.text((70,100),'Thermal evidence, frame by frame',font=font(54,True),fill='#f0f1df')
        draw.text((70,181),'RECORDED FLIGHT  /  SAVED MODEL LABELS  /  NO NEW INFERENCE',font=font(19),fill='#93b5aa')
        if available:
            frame = annotated(nearest).resize((960,768),Image.Resampling.NEAREST)
            image.paste(frame,(70,245)); used.append(nearest['id'])
        else:
            draw.rectangle((70,245,1030,1013),fill='#132b27')
            draw.text((145,590),'No labelled frame at this source time',font=font(24),fill='#f0f1df')
        draw.text((1110,270),'SOURCE TIME',font=font(20,True),fill='#93b5aa')
        draw.text((1110,315),f'{offset:05.1f} seconds',font=font(47,True),fill='#f0f1df')
        draw.text((1110,445),'MODEL CANDIDATES',font=font(20,True),fill='#93b5aa')
        draw.text((1110,490),str(len(nearest.get('boxes',[]))) if available else '—',font=font(75,True),fill='#bcf29e')
        for y,line in enumerate(['Confidence threshold: 0.60','Machine labels, not human review.','Candidates are not verified identities.','','Playback: 5 saved frames / second.','Missing labels remain visible gaps.']):
            draw.text((1110,635+y*42),line,font=font(22),fill='#d1dfd3')
        draw.text((1110,964),rows[0]['source']['session_id'],font=font(24),fill='#93b5aa')
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait(): raise RuntimeError('Thermal video encoding failed')
    working.replace(video)
    evidence = {'captured_at':stamp,'source':source,'start_offset_s':start,'duration_s':20,'sample_playback_fps':5,'frame_ids':sorted(set(used)),
                'selection':'One training-split source, window selected for saved candidate density; illustrative, not a representative accuracy benchmark',
                'missing_frame_policy':'Explicit blank frame when no saved labelled frame is within 0.14 s','model':chosen['model'],'human_reviewed':False,'selected_image_sha256':chosen['sha256']}
    (OUT/'thermal-video-provenance.json').write_text(json.dumps(evidence,indent=2))
    counts = state['inputs']['counts']; control = state['controller']; queue = control.get('queue_counts',{})
    native = counts['native_labels']; accepted = control.get('stats',{}).get('accepted_pairs',0)
    completed = queue.get('completed',0); active = queue.get('running',0)
    sections = [
      ('01 / THE QUESTION','Can a thermal survey agent<br>learn to gather better evidence?',
       'Real flight data. A physics-informed training world. A model whose progress must survive held-out evaluation.',
       '<div class="hero-image"><img src="assets/25-thermal-candidate.png"><span>REAL THERMAL FRAME · MODEL CANDIDATES</span></div>',
       'We are building an evidence pipeline for thermal drone surveys. Our question is whether stronger supervision and physically grounded synthetic observations can improve detection and survey decisions.'),
      ('02 / THE EVIDENCE','Start with the ground<br>we actually flew over.',
       'Source frames, original responses and label provenance stay together. Whole recordings and zones retain their held-out splits.',
       f'<div class="stats"><div><strong>{native:,}</strong><span>native frames indexed</span></div><div><strong>{counts["manual_anchor_train"]:,}</strong><span>reviewed training anchors</span></div><div><strong>{counts["zip_screened_backgrounds"]:,}</strong><span>screened ZIP background crops</span></div></div><div class="pair"><figure><img src="assets/25-thermal-original.png"><figcaption>Original recorded frame</figcaption></figure><figure><img src="assets/25-thermal-candidate.png"><figcaption>Saved teacher candidates · confidence ≥ 0.60</figcaption></figure></div>',
       'The thermal archive is sampled at five frames per second. We cache the teacher response before parsing it, retain hashes and splits, and archive the labels and logs. Machine labels remain explicitly marked as machine labels.'),
      ('03 / THE PIPELINE','Evidence arrives.<br>The next stage starts.',
       'Gemini Robotics ER 2 supplies cached thermal labels. Gemma 12B is the model we train. Frozen batches connect the two.',
       '<div class="flow"><div><b>01</b><h3>Record</h3><p>Thermal frames<br>Source timestamps</p></div><i>→</i><div><b>02</b><h3>Label</h3><p>Saved responses<br>Traceable candidates</p></div><i>→</i><div><b>03</b><h3>Learn</h3><p>Supervised tuning<br>GRPO experiments</p></div><i>→</i><div><b>04</b><h3>Evaluate</h3><p>Frozen development set<br>Promotion gate</p></div></div><div class="callout">Physics → numerical admission → auxiliary synthetic crops. This branch remains gated.</div>',
       'This is a streaming system. Eligible new data releases a frozen training batch without waiting for the whole archive. Physical outputs follow a separate admission gate. The complete synthetic study has not finished.'),
      ('04 / THE PHYSICS','Simulate the full day.<br>Earn numerical acceptance.',
       'Coupled heat and moisture, atmospheric forcing and evaporation. Refinement checks test whether the result depends on discretization.',
       f'<div class="physical"><img src="assets/physics-simulation-frame.png"><div><div class="pill">EXPERIMENTAL SOLVER OUTPUT</div><h3>24 virtual hours</h3><p>Grid · time step · domain<br>Conservation · interfaces</p><strong>{accepted}</strong><p>accepted physical pairs at capture</p></div></div>',
       'The previous mesh failed convergence, so we preserved that result and refined the heated soil layers. Corrected full-day runs are now executing. This is numerical qualification, not field validation, and unaccepted fields remain outside training.'),
      ('05 / THE EXPERIMENT','Four GPUs.<br>A useful comparison.',
       'Two matched 964-image datasets × three seeds. One supervised epoch per run; identical negative backgrounds and development evaluation.',
       f'<div class="arms"><div><span>ARM A</span><h3>Reviewed anchors</h3><p>482 positive anchors<br>482 screened backgrounds</p></div><div><span>ARM B</span><h3>Native-label mixture</h3><p>241 reviewed + 241 native positives<br>The same 482 backgrounds</p></div></div><div class="stats compact"><div><strong>{active}</strong><span>jobs running at capture</span></div><div><strong>{queue.get("queued",0)}</strong><span>jobs queued</span></div><div><strong>{completed}</strong><span>earlier stream runs completed</span></div></div>',
       'The first streaming runs produced a weak and often flat reinforcement signal. We are now isolating label source and supervised training coverage before drawing conclusions about RL. The active comparison is supervised fine tuning, not another claimed RL gain.'),
      ('06 / THE LIVE SYSTEM','Every result has a trail.',
       'Actual resource measurements, solver clocks, optimizer logs, source images and failure records are visible in one place.',
       '<img class="dashboard" src="assets/27-live-dashboard.png">',
       'The dashboard reports measured utilization rather than allocated hardware. The workflows persist beyond this laptop session. We keep failed numerical checks and negative learning results so the presentation can be audited.'),
      ('07 / THE NEXT RESULT','An experiment in progress.<br>A claim we can test.',
       'Does stronger supervision improve held-out detection? Can admitted physical variation improve robustness beyond the recorded scene?',
       '<div class="next"><div><span>BUILT & RUNNING</span><h3>Data → labels → training</h3><p>Archived responses, frozen datasets,<br>four active GPU jobs and physical checks.</p></div><div><span>STILL TO ESTABLISH</span><h3>Generalization</h3><p>Held-out gains, accepted synthetic fields,<br>and calibrated transfer to real flights.</p></div></div><div class="callout">A lower training loss is not detection accuracy. A model candidate is not a verified mine.</div>',
       'Our deliverable is a reproducible experimental system and its measured results. We do not yet claim that RL improves detection, that the synthetic branch is accepted, or that candidate detections are independently verified mines.')]
    css = '''*{box-sizing:border-box}html{scroll-snap-type:y mandatory}body{margin:0;background:#071715;color:#f2f0de;font:24px Arial,sans-serif}section{position:relative;width:100%;height:100vh;min-height:800px;padding:65px 80px;overflow:hidden;scroll-snap-align:start;background:radial-gradient(ellipse at 90% 10%,#183a30 0,transparent 55%),#091b18}header{display:flex;justify-content:space-between;font-size:16px;letter-spacing:3px;color:#baf19e}h1{font-size:68px;line-height:1.05;letter-spacing:-3px;font-weight:500;margin:36px 0 20px;max-width:1100px}p.lead{max-width:1100px;font-size:25px;line-height:1.45;color:#b4c9bb;margin:0 0 28px}footer{position:absolute;bottom:25px;left:80px;right:80px;display:flex;justify-content:space-between;color:#8ba99b;font-size:13px;letter-spacing:.7px}img{object-fit:contain}.hero-image{position:absolute;right:80px;bottom:80px;width:720px;height:440px;background:#142c26;border:1px solid #45644d;padding:16px;transform:rotate(-2deg)}.hero-image img{width:100%;height:94%;object-fit:cover;object-position:center 45%;filter:none}.hero-image span{font-size:12px;letter-spacing:2px;color:#baf19e}.stats{display:flex;gap:60px;margin:25px 0}.stats strong{display:block;font-size:60px;letter-spacing:-3px;color:#c8f09b}.stats span{font-size:17px;color:#b4c9bb}.pair{display:flex;gap:28px;height:380px;margin-top:20px}figure{margin:0;flex:1;background:#132a24;padding:12px;display:flex;flex-direction:column;align-items:center}figure img{height:calc(100% - 35px);max-width:100%}figcaption{font-size:16px;margin-top:10px;color:#b4c9bb}.flow{display:flex;gap:22px;align-items:center;margin-top:80px}.flow div{flex:1;border-top:1px solid #68835b;background:#132b24;padding:30px 25px;height:230px}.flow b{color:#c8f09b;font-size:18px}.flow h3{font-size:36px;margin:18px 0 12px}.flow p{font-size:20px;line-height:1.5;color:#b4c9bb}.flow i{font-style:normal;color:#c8f09b}.callout{padding:24px 28px;border-left:3px solid #c8f09b;background:#193529;font-size:21px;color:#d6e5c8;margin-top:30px}.physical{display:flex;gap:45px;align-items:center;margin-top:30px}.physical img{width:68%;height:480px;background:#0b1518}.physical h3{font-size:38px}.physical p{color:#b4c9bb;line-height:1.5;font-size:21px}.physical strong{font-size:70px;color:#c8f09b}.pill{border:1px solid #bb9968;border-radius:30px;font-size:12px;padding:12px;color:#efd59c}.arms,.next{display:flex;gap:32px;margin-top:50px}.arms>div,.next>div{flex:1;padding:32px 38px;border:1px solid #46664b;background:#132a24}.arms span,.next span{font-size:16px;letter-spacing:2px;color:#baf19e}.arms h3,.next h3{font-size:34px}.arms p,.next p{font-size:24px;line-height:1.5;color:#b4c9bb}.compact{margin-top:35px}.compact strong{font-size:48px}.dashboard{width:100%;height:540px;object-fit:cover;object-position:top;border:1px solid #46664b}section:first-child h1{max-width:1100px;font-size:84px}section:first-child p{max-width:590px}a{color:#c8f09b}.assets{position:fixed;bottom:8px;right:14px;z-index:4;font-size:11px} @media print{@page{size:1600px 1000px;margin:0}html{scroll-snap-type:none}section{height:1000px;width:1600px;min-height:0;page-break-after:always;break-inside:avoid;print-color-adjust:exact;-webkit-print-color-adjust:exact}.assets{display:none}}'''
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AeroRozum · Live research briefing</title><style>'+css+'</style><body>'
    notes = ['# AeroRozum · live research briefing', '', f'Evidence captured: {stamp}', '', 'All counts below describe this snapshot. Training and solver jobs continue.', '']
    for i,(eyebrow,title,lead,body,note) in enumerate(sections):
        document += f'<section id="slide-{i+1}"><header><b>AEROROZUM</b><span>{eyebrow}</span></header><h1>{title}</h1><p class="lead">{lead}</p>{body}<footer><span>Research in progress · {esc(stamp[:16])} UTC</span><span>{i+1:02} / {len(sections):02}</span></footer></section>'
        notes += ['## '+title.replace('<br>',' '),'',note,'']
    document += '<div class="assets"><a href="live-briefing.pdf">PDF</a> · <a href="assets/thermal-evidence-20s.mp4">Thermal video</a> · <a href="assets/dashboard-walkthrough.mp4">Dashboard video</a> · <a href="live-briefing-notes.md">Speaker notes</a></div><script>addEventListener("keydown",e=>{let s=[...document.querySelectorAll("section")],i=s.findIndex(v=>v.getBoundingClientRect().bottom>100);if(["ArrowRight","PageDown"," "].includes(e.key)){e.preventDefault();s[Math.min(i+1,s.length-1)].scrollIntoView()}if(["ArrowLeft","PageUp"].includes(e.key)){e.preventDefault();s[Math.max(0,i-1)].scrollIntoView()}})</script></body></html>'
    (OUT/'live-briefing.html').write_text(document)
    notes += ['## Suggested 90-second narration','','We start from the ground our drone actually flew over. Thermal video gives us valuable evidence, but not ready-made truth. Our pipeline samples those recordings, saves the model response and provenance, and creates traceable training data. Gemini Robotics ER 2 is the teacher; Gemma is the model we train.','','In parallel, we simulate a full heating and cooling day with coupled heat and moisture physics. Numerical checks decide whether those fields can enter training. The earlier mesh failed a convergence check, and the refined runs are now being tested.','','Nine earlier streaming runs are complete. The reinforcement signal was weak, so our current six-run comparison isolates reviewed supervision versus a mixture with native machine labels. Four GPUs execute the comparison while the CPUs run physics qualification.','','The dashboard shows real optimizer logs, physical clocks and archived evidence. What matters next is a held-out improvement. We are presenting a working experimental system, its failures and its next measurable result—not a claimed detection breakthrough.','','## Video provenance','','The thermal clip is an illustrative candidate-rich 20-second window from one training recording, not an accuracy sample. It uses saved labelled frames, preserves the source-time clock and displays missing-frame gaps. Candidate boxes are model outputs, not independently verified mine identities. See thermal-video-provenance.json.']
    (OUT/'live-briefing-notes.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps({'captured_at':stamp,'native_frames':native,'training_jobs_running':active,'thermal_clip_source':source,'thermal_clip_window_s':[start,start+20],'deck':str(OUT/'live-briefing.html')},indent=2))

if __name__ == '__main__': main()
