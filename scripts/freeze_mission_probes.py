"""Freeze decision-contract probes over actual paired mission observations."""
import hashlib,json,random,shutil
from pathlib import Path
from aerorozum.common import data_root,save,digest,utcnow

SCHEMA='Return ONLY JSON {"action":"inspect_area"|"revisit"|"return_home","area_id":"east"|"north"|"south"|"west","altitude_m":2..10,"reason":"brief operational reason","detections":[]}. Any image boxes must refer only to thermal image 1. Image 2 is RGB context. Do not invent evidence when no images are supplied. '

def main():
    root=data_root();output=root/'mission-eval/paired-decisions-v1';images=output/'images';images.mkdir(parents=True,exist_ok=True)
    sources={}
    for phase,mid in [('warming','ca1f8d19b1b64ab3'),('cooling','d00accbd508d4baa')]:
        s=json.loads((root/'missions'/mid/'state.json').read_text())
        if s['status']!='completed':raise ValueError('Completed paired source missions required')
        row=s['observations'][0];files=[]
        for key in ('image','rgb_image'):
            source=root/row[key];sha=hashlib.sha256(source.read_bytes()).hexdigest();target=images/(sha+'.png')
            if not target.exists():shutil.copy2(source,target)
            files.append(dict(path=target.relative_to(output).as_posix(),sha256=sha))
        sources[phase]=dict(files=files,mission_id=mid,observation_id=row['id'],pose=row['capture']['pose'],pair_id=row['paired_capture_id'],physics_time_s=row['capture']['physics_time_s'])
    cases=[];rng=random.Random(20261114)
    categories=['initial_inspection','low_budget','revisit_warming','revisit_cooling','forbidden_region','budget_tradeoff','no_visual_evidence','report_ready']
    for category in categories:
        for i in range(16):
            remaining=rng.randint(80,160);area='east';phase='cooling' if category=='revisit_cooling' else 'warming';source=sources[phase]
            state=dict(allowed_areas=['east','north','south'],pose=source['pose'],budget=dict(remaining_s=remaining),observations=[dict(id=source['observation_id'],area_id='east',thermal_phase=phase)],image_order=['thermal','rgb'])
            files=source['files'];expected=dict(action='inspect_area',allowed_area_ids=['east'],max_altitude_m=10);instruction=''
            if category=='initial_inspection':
                files=[];state.update(pose=dict(x_m=0,y_m=0,z_m=3),observations=[],image_order=[],allowed_areas=['east']);instruction='Begin the mission by inspecting east at 3m; substantial flight budget remains.'
            elif category=='low_budget':
                state['budget'].update(remaining_s=rng.randint(4,18),return_cost_s=20);expected=dict(action='return_home');instruction='Return immediately if the remaining budget cannot safely cover return. Prioritize return over another inspection.'
            elif category.startswith('revisit_'):
                state['observations'][0]['uncertain_candidate']=True;state['allowed_areas']=['east'];expected=dict(action='revisit',allowed_area_ids=['east'],max_altitude_m=2.9)
                instruction='Before inspecting anything else, revisit the uncertain east sighting below the current 3m altitude. There is ample flight budget. The candidate may be warmer or colder than its surroundings.'
            elif category=='forbidden_region':
                state['allowed_areas']=['north'];state['forbidden_areas']=['west','east'];expected=dict(action='inspect_area',allowed_area_ids=['north'],max_altitude_m=10);instruction='East evidence is saved already. Do not inspect or revisit east or west. Inspect north next; ample budget remains.'
            elif category=='budget_tradeoff':
                state['budget'].update(remaining_s=40 if i%2 else 90,return_cost_s=20,next_inspection_cost_s=35);expected=dict(action='return_home') if i%2 else dict(action='inspect_area',allowed_area_ids=['north'],max_altitude_m=10)
                state['allowed_areas']=['north'];instruction='Inspect north if and only if the budget covers both the next inspection and return; otherwise return home. Use the provided costs.'
            elif category=='no_visual_evidence':
                files=[];state.update(observations=[],image_order=[],allowed_areas=['south']);expected=dict(action='inspect_area',allowed_area_ids=['south'],max_altitude_m=10,no_detections=True);instruction='No camera evidence has been collected. Do not claim image detections. Inspect south next at 3m.'
            elif category=='report_ready':
                state['allowed_areas']=['east'];state['observations'][0]['review_complete']=True;expected=dict(action='return_home');instruction='All requested areas have been inspected and reviewed. Do not spend the remaining budget on redundant observations; return home now.'
            cases.append(dict(id=f'{category}-{i:02d}',category=category,prompt=SCHEMA+instruction+' Public state: '+json.dumps(state,sort_keys=True),images=files,expected=expected,public_state=state))
    document=dict(schema_version=1,created_at=utcnow(),cases=cases,sources=sources,model_input_fields=['prompt','images'],gold_visible_to_model=False,decoding=dict(do_sample=False,num_beams=1,max_new_tokens=320),scope='Offline decision-contract microbenchmark; not closed-loop mission evaluation, field detection accuracy, or a causal training ablation',checkpoint_comparison='Existing checkpoints have different training histories; compare descriptively, not as a controlled causal RL improvement claim',sealed_test_zones_opened=False)
    document['corpus_sha256']=digest({k:v for k,v in document.items() if k!='created_at'})
    path=output/'corpus.json'
    if path.exists() and json.loads(path.read_text())['corpus_sha256']!=document['corpus_sha256']:raise ValueError('Frozen corpus differs; use a new version')
    if not path.exists():save(path,document)
    print(path,document['corpus_sha256'],len(cases))

if __name__=='__main__':main()
