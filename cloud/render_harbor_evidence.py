"""Presentation replay from verified captured frames, never invented viewpoints."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

BG='#10191d';CARD='#19262b';LINE='#30434a';TEXT='#eef5f4';MUTED='#a5b9bf';ACCENT='#72e8c0';BOX='#70ccff'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


def render(trial,output):
    trial=Path(trial);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((trial/'verifier/ledger.json').read_text());result=ledger['result']
    metadata=json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else {}
    agent=metadata.get('config',{}).get('agent',{});model=agent.get('model_name') or agent.get('name') or 'reference policy'
    if model=='oracle':model='Public-image reference policy'
    observations=sorted(ledger['observations'].values(),key=lambda r:r['sim_time_s'])
    if not observations:raise ValueError('No captured evidence to render')
    evidence={}
    for i,candidate in enumerate((ledger.get('report') or {}).get('candidates',[]),1):
        for item in candidate.get('evidence',[]):evidence.setdefault(item['observation_id'],[]).append((i,item['bbox']))
    receipts=[]
    for number,row in enumerate(observations):
        frame=Image.new('RGB',(1920,1080),BG);draw=ImageDraw.Draw(frame)
        draw.rounded_rectangle((38,34,92,88),radius=16,fill=ACCENT);draw.text((51,40),'A',font=font(32,True),fill=BG)
        draw.text((110,36),'AeroRozum',font=font(31,True),fill=TEXT)
        draw.text((111,76),'HARBOR / INVESTIGATION BENCHMARK',font=font(14),fill=MUTED)
        draw.text((38,126),'Evidence that survives the flight.',font=font(46,True),fill=TEXT)
        draw.text((40,190),f'{model}  ·  capture {number+1:02d}/{len(observations):02d}  ·  sim t={row["sim_time_s"]:.2f}s',font=font(20),fill=MUTED)
        for kind,x in [('thermal',38),('rgb',754)]:
            item=next(i for i in row['images'] if i['kind']==kind)
            path=trial/'verifier/evidence'/Path(item['filename']).name;payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('Camera evidence hash changed')
            image=Image.open(path).convert('RGB');imdraw=ImageDraw.Draw(image)
            if kind=='thermal':
                for index,b in evidence.get(row['id'],[]):
                    values=[b[k] for k in ['x_min','y_min','x_max','y_max']]
                    if not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1000 for v in values):continue
                    a,c,e,f=values;bounds=(a/1000*image.width,c/1000*image.height,e/1000*image.width,f/1000*image.height)
                    if bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:continue
                    imdraw.rectangle(bounds,outline=BOX,width=2);imdraw.text((bounds[0],max(0,bounds[1]-19)),f'C{index:02d}',font=font(15,True),fill=BOX)
            draw.rounded_rectangle((x,245,x+688,847),radius=16,fill=CARD)
            frame.paste(image.resize((664,531)),(x+12,292))
            draw.text((x+18,258),'THERMAL / SIMULATED RESPONSE' if kind=='thermal' else 'RGB / NATIVE GAZEBO CAMERA',font=font(16,True),fill=ACCENT if kind=='thermal' else TEXT)
            receipts.append(dict(observation_id=row['id'],kind=kind,sha256=item['sha256']))
        left,top=1474,245
        draw.rounded_rectangle((left,top,1882,847),radius=16,fill=CARD)
        draw.text((left+22,top+20),'CAPTURE POSITIONS',font=font(18,True),fill=TEXT)
        def point(x,y):return left+30+(x+6)/24*346,top+100+(18-y)/24*346
        for v in range(-5,20,5):
            draw.line([point(v,-6),point(v,18)],fill=LINE);draw.line([point(-6,v),point(18,v)],fill=LINE)
        for name,rect in [('EAST',(5,-5,15,5)),('NORTH',(-5,7,5,17))]:
            a,b,c,d=rect;draw.rectangle([point(a,d),point(c,b)],outline=ACCENT,width=2);draw.text(point(a,d),name,font=font(13,True),fill=ACCENT)
        hx,hy=point(0,0);draw.ellipse((hx-6,hy-6,hx+6,hy+6),fill=TEXT);draw.text((hx+8,hy),'HOME',font=font(12),fill=TEXT)
        for j,r in enumerate(observations):
            px,py=point(r['pose']['x_m'],r['pose']['y_m']);color=ACCENT if j==number else MUTED
            draw.ellipse((px-5,py-5,px+5,py+5),fill=color);draw.text((px+6,py-14),str(j+1),font=font(12),fill=color)
        p=row['pose'];draw.text((left+22,top+486),f'E {p["x_m"]:5.2f}m   N {p["y_m"]:5.2f}m',font=font(17),fill=TEXT)
        draw.text((left+22,top+520),f'Altitude {p["z_m"]:.2f}m',font=font(17),fill=MUTED)
        draw.text((left+22,top+555),'Points are actual capture poses.',font=font(14),fill=MUTED)
        cards=[('VERIFIER REWARD',f'{result["reward"]:.3f} / 1'),('REGION COVERAGE',f'{result["qualified_coverage"]*100:.1f}%'),('MATCHED / EXTRA CLAIMS',f'{result["true_positives"]} / {result["false_positives"]}'),('FLIGHT / BUDGET',f'{result["flight_elapsed_s"]:.1f} / {result["budget_s"]:.0f}s')]
        for i,(label,value) in enumerate(cards):
            x=38+i*467;draw.rounded_rectangle((x,871,x+443,988),radius=13,fill=CARD)
            draw.text((x+20,887),label,font=font(14),fill=MUTED);draw.text((x+20,918),value,font=font(30,True),fill=ACCENT if i==0 else TEXT)
        draw.text((40,1012),'Saved evidence replay. Blue boxes are submitted claims. Inserted synthetic signatures; field validation not performed.',font=font(17),fill=MUTED)
        draw.text((40,1044),f'Capture {row["id"]}  ·  images verified against the sealed ledger',font=font(13),fill=MUTED)
        frame.save(output/f'frame-{number:03d}.png')
    index=max(range(len(observations)),key=lambda n:len({i for i,b in evidence.get(observations[n]['id'],[])}))
    Image.open(output/f'frame-{index:03d}.png').save(output/'harbor-evidence.png')
    encoder=shutil.which('ffmpeg')
    if encoder is None:
        import imageio_ffmpeg
        encoder=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([encoder,'-y','-loglevel','error','-framerate','1/2','-i',str(output/'frame-%03d.png'),'-c:v','libx264','-threads','2','-r','24','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'harbor-evidence-replay.mp4')],check=True)
    (output/'provenance.json').write_text(json.dumps(dict(trial=str(trial),ledger_sha256=hashlib.sha256((trial/'verifier/ledger.json').read_bytes()).hexdigest(),result=result,model=model,images=receipts,annotation='Submitted candidate boxes; no private-reference overlay',video='2 seconds per captured pair, chronological cuts; not a continuous-flight recording'),indent=2))
    print(output/'harbor-evidence-replay.mp4')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--output',required=True);a=p.parse_args();render(a.trial,a.output)
