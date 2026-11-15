"""Render only retained, telemetry-qualified real thermal evidence; no API calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/opt/aerorozum');DATA=ROOT/'airborne-labeler/artifacts'
LABELS=DATA/'labelling/robotics-er2-high-thinking-v1'
PUBLIC=ROOT/'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'


def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def main():
    from aerorozum.common import save,utcnow
    groups=defaultdict(list)
    for path in (LABELS/'results').glob('*.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='invalid_model_response' or not isinstance(row.get('detections'),list):continue
        if not row.get('airborne_eligible') or row.get('eligibility',{}).get('status')!='eligible':continue
        groups[(row['source']['object'],str(row['source']['generation']))].append(row)
    eligible=[rows for rows in groups.values() if len(rows)>=80 and sum(bool(r.get('boxes')) for r in rows)>=5]
    if not eligible:raise ValueError('Wait for at least 80 qualified samples and five candidate-bearing frames in one source')
    rows=max(eligible,key=lambda rows:(sum(bool(r.get('boxes')) for r in rows),len(rows)))
    rows.sort(key=lambda r:r['source_pts_s'])
    # This is explicitly an illustrative candidate-rich segment, not an
    # unbiased accuracy sample. Choose its earliest 20-second interval.
    start=next(r['source_pts_s'] for r in rows if r.get('boxes'))
    rows=[r for r in rows if start<=r['source_pts_s']<=start+20]
    if len(rows)<40:raise ValueError('The selected interval does not yet contain enough retained evidence')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    bindings=[]
    with tempfile.TemporaryDirectory(prefix='aerorozum-evidence-',dir=ROOT/'archives') as temporary:
        temporary=Path(temporary);lines=[]
        for i,row in enumerate(rows):
            raw=Path(row['image']);raw=raw if raw.is_absolute() else DATA/raw
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Original image differs')
            with Image.open(raw) as source:image=source.convert('RGB')
            if image.size!=(640,512):raise ValueError('Preserve the native 640 by 512 frame')
            canvas=Image.new('RGB',(1280,704),'#0b1419');canvas.paste(image,(0,64));canvas.paste(image,(640,64));draw=ImageDraw.Draw(canvas)
            draw.text((18,14),'AEROROZUM / ORIGINAL THERMAL',font=font(23),fill='#e5f2f1')
            draw.text((658,14),'CLASSIFIED MODEL 2.0 / HIGH',font=font(23),fill='#6ce5cb')
            for detection in row['detections']:
                box=detection['bbox'];confidence=detection['confidence'];color='#6ce5cb' if confidence>=.6 else '#ffbf70'
                x1=640+box['x_min']*.64;y1=64+box['y_min']*.512;x2=640+box['x_max']*.64;y2=64+box['y_max']*.512
                draw.rectangle((x1,y1,x2,y2),outline=color,width=3)
                draw.text((x1,max(64,y1-20)),f'{confidence:.2f}',font=font(17),fill=color,stroke_width=1,stroke_fill='#102027')
            relative=row['source_pts_s']-start
            draw.text((18,592),f"Source time {row['source_pts_s']:.1f} s  |  clip +{relative:.1f} s  |  frame {row['frame_index']}",font=font(20),fill='#e5f2f1')
            draw.text((18,625),f"{len(row['boxes'])} accepted candidate(s)   |   {len(row['uncertain_boxes'])} lower-confidence signature(s)   |   fresh IN_AIR telemetry",font=font(18),fill='#9eb6bb')
            draw.text((18,663),'Recorded 5-FPS evidence samples. Boxes are model proposals; physical mine identity is unverified.',font=font(17),fill='#9eb6bb')
            filename=temporary/f'{i:05d}.png';canvas.save(filename)
            lines.append('file '+filename.as_posix())
            duration=(rows[i+1]['source_pts_s']-row['source_pts_s']) if i+1<len(rows) else .2
            if not 0<duration<=2:raise ValueError('Do not hide long missing-evidence intervals in a continuous clip')
            lines.append('duration '+str(duration))
            bindings.append(dict(id=row['id'],source_pts_s=row['source_pts_s'],sha256=row['sha256'],cache_key=row['cache_key']))
        lines.append('file '+(temporary/f'{len(rows)-1:05d}.png').as_posix())
        concat=temporary/'frames.txt';concat.write_text('\n'.join(lines)+'\n')
        output=PUBLIC/'high-thinking-flight.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-r','25','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True,timeout=300)
    receipt=dict(at=utcnow(),source=rows[0]['source'],frames=len(rows),bindings=bindings,
        selection='Illustrative candidate-rich source; earliest 20-second interval beginning at a retained accepted candidate. Not an accuracy sample.',
        original_source_timing_preserved=True,model_inference_called=False,source_pixels_changed=False,
        rendered_side_by_side=True,physical_identity_verified=False,sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    save(PUBLIC/'high-thinking-flight.json',receipt);(PUBLIC/'high-thinking-flight.json').chmod(0o644)
    print(json.dumps({k:v for k,v in receipt.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
