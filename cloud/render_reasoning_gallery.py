"""Render recorded predictions on original development pixels; no API calls."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path('/opt/aerorozum')
STUDY=ROOT/'artifacts/overnight-20261115T000942/robotics-reasoning-v1'
PUBLIC=ROOT/'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def main():
    from aerorozum.common import save
    from aerorozum.label_schema import normalize_thermal_output
    protocol=json.loads((STUDY/'protocol.json').read_text());rows=protocol['images']
    manual=[r for r in rows if r.get('zone')=='zone7' and r.get('boxes')]
    cases=[('manual-single',min(manual,key=lambda r:(len(r['boxes']),r['id']))),
           ('manual-complex',max(manual,key=lambda r:(len(r['boxes']),r['id']))),
           ('machine-screened-zero',next(r for r in rows if not r.get('boxes')))]
    records={}
    for p in (STUDY/'results').glob('*.json'):
        r=json.loads(p.read_text())
        if r['replica']==0:records[(r['id'],r['prompt_version'],r['thinking_level'])]=r
    PUBLIC.mkdir(parents=True,exist_ok=True);gallery=[]
    for name,row in cases:
        canvas=Image.new('RGB',(1480,990),'#0c1519');d=ImageDraw.Draw(canvas)
        d.text((24,18),'AEROROZUM / REAL THERMAL PIXELS',font=font(25),fill='#eef7f5')
        d.text((24,58),name.replace('-',' ').upper()+'  |  fixed example, replica 1',font=font(17),fill='#8ca9ad')
        for iy,prompt in enumerate(protocol['prompts']):
            for ix,level in enumerate(['minimal','medium','high']):
                x,y=24+ix*488,105+iy*415;record=records.get((row['id'],prompt,level))
                d.rounded_rectangle((x,y,x+464,y+395),radius=14,fill='#15242b')
                d.text((x+14,y+12),('V2' if 'v2' in prompt else 'V3')+' / '+level.upper(),font=font(19),fill='#63ddc5')
                with Image.open(ROOT/'artifacts'/row['image']) as im:image=im.convert('RGB')
                image=ImageOps.contain(image,(436,292));left=x+14+(436-image.width)//2;top=y+45
                canvas.paste(image,(left,top));d=ImageDraw.Draw(canvas)
                if record and record.get('valid_output'):
                    output,_=normalize_thermal_output(record['model_output'])
                    for detection in output['detections']:
                        b=detection['bbox'];score=detection['confidence'];color='#63ddc5' if score>=.6 else '#ffbf70'
                        a=left+b['x_min']/1000*image.width;c=top+b['y_min']/1000*image.height
                        z=left+b['x_max']/1000*image.width;v=top+b['y_max']/1000*image.height
                        d.rectangle((a,c,z,v),outline=color,width=3)
                        d.text((a,max(top,c-15)),f'{score:.2f}',font=font(13),fill=color,stroke_width=1,stroke_fill='#101619')
                    score=record['score'];usage=record.get('usage',{})
                    d.text((x+14,y+342),f"TP {score['tp']}   FP {score['fp']}   FN {score['fn']}   |   F2 {score['f2']:.3f}",font=font(15),fill='#d2e5e2')
                    d.text((x+14,y+368),f"{record['elapsed_s']:.1f}s   |   thought tokens: {usage.get('thoughts_token_count',0) or 0}",font=font(13),fill='#8ca9ad')
                else:d.text((x+14,y+342),'Awaiting a retained valid response',font=font(14),fill='#ffbf70')
        note='Reference annotations: '+row.get('provenance','unknown') if row.get('boxes') else 'Zero-candidate machine screening is not verified mine-free ground.'
        d.text((24,950),note+'  Green >= 0.60; amber < 0.60. Development only.',font=font(14),fill='#8ca9ad')
        canvas.save(PUBLIC/(name+'.png'))
        gallery.append(dict(image=name+'.png',id=row['id'],source_sha256=row['sha256'],split='validation',reference_provenance=row.get('provenance'),gold_boxes=row.get('boxes'),case_selection='Predefined first sorted single/multiple-box manual example and first machine-screened zero example'))
    save(PUBLIC/'gallery.json',gallery)
    (PUBLIC/'gallery.json').chmod(0o644)


if __name__=='__main__':main()
