"""Build a static evidence browser from saved labels and accepted image manifests."""
import collections,hashlib,json,pathlib,shutil,subprocess
from PIL import Image,ImageDraw,ImageFont

ROOT=pathlib.Path('/opt/aerorozum');DATA=ROOT/'airborne-labeler/artifacts'
MEDIA=ROOT/'deployment/artifacts/presentation/evidence-browser'
PAGE=ROOT/'deployment/releases/coordinate-agent-v3/docs/presentation/evidence-browser'
def pts(row):return float(row.get('source_pts_s',row.get('timestamp_s',0)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def render(rows):
    identity=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest()[:20]
    filename='clip-'+identity;video=MEDIA/(filename+'.mp4');poster=MEDIA/(filename+'.jpg')
    info=dict(id=identity,session=rows[0]['source']['session_id'],segment=pathlib.PurePosixPath(rows[0]['source']['object']).name,
        start_s=pts(rows[0]),end_s=pts(rows[-1]),frames=len(rows),positive_frames=sum(bool(r.get('boxes')) for r in rows),
        video='/artifacts/presentation/evidence-browser/'+video.name,poster='/artifacts/presentation/evidence-browser/'+poster.name,
        source=rows[0]['source'],prompt_versions=sorted({r.get('label_prompt_version','unknown') for r in rows}),
        frame_ids=[r['id'] for r in rows],source_times_s=[pts(r) for r in rows],new_inference_calls=0)
    if video.exists() and poster.exists():return info
    import imageio_ffmpeg
    temp=MEDIA/(filename+'.rendering.mp4');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x592','-r','5','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    best=-1;cover=None
    try:
        for row in rows:
            path=DATA/row['image']
            if sha(path)!=row['sha256']:raise ValueError('Source image checksum mismatch')
            with Image.open(path) as source:image=source.convert('RGB').resize((640,512))
            canvas=Image.new('RGB',(640,592),(15,23,28));canvas.paste(image,(0,80));draw=ImageDraw.Draw(canvas)
            for det in row.get('detections',[]):
                box=det['bbox'];confidence=det.get('confidence',0);color=(80,245,175) if confidence>=.6 else (255,198,70)
                x0=box['x_min']*.64;y0=box['y_min']*.512+80;x1=box['x_max']*.64;y1=box['y_max']*.512+80
                label=det.get('label','Candidate')
                if 'mine' in label.lower() or 'candidate' in label.lower():label='Candidate' if confidence>=.6 else 'Uncertain'
                label=label[:18]+f' {confidence:.2f}';ty=max(80,y0-23)
                draw.rectangle((x0,y0,x1,y1),outline=color,width=3);draw.rectangle((x0,ty,min(640,x0+195),ty+22),fill=(15,23,28));draw.text((x0+3,ty+2),label,font=font,fill=color)
            count=len(row.get('boxes',[]))
            draw.text((12,6),'THERMAL FLIGHT | Saved Gemini Robotics labels',font=font,fill='white')
            draw.text((12,31),f"{info['session']} | source {pts(row):.1f}s | {count} candidates",font=font,fill=(180,205,210))
            draw.text((12,55),'5 fps sampled replay | green >= 0.60 | amber < 0.60',font=font,fill=(180,205,210))
            encoder.stdin.write(canvas.tobytes())
            if count>best:best=count;cover=canvas.copy()
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
        temp.replace(video);cover.save(poster,quality=90)
    except Exception:
        encoder.kill();encoder.wait();temp.unlink(missing_ok=True);raise
    (MEDIA/(filename+'.json')).write_text(json.dumps(info,indent=2));return info

def main():
    MEDIA.mkdir(parents=True,exist_ok=True);PAGE.mkdir(parents=True,exist_ok=True);groups=collections.defaultdict(list)
    for path in (DATA/'labelling/robotics-er2-airborne/labels/results').glob('*.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='invalid_model_response' or not row.get('image') or not (DATA/row['image']).is_file():continue
        groups[(row['source']['session_id'],row['source']['object'],str(row['source']['generation']))].append(row)
    by_session=collections.defaultdict(list)
    for key,rows in sorted(groups.items()):
        rows.sort(key=pts)
        if len(rows)>=50:by_session[key[0]].append(rows)
    chosen=[]
    for i in range(max(map(len,by_session.values()),default=0)):
        for session,segments in sorted(by_session.items()):
            if i<len(segments):chosen.append(segments[i])
    clips=[]
    for index,rows in enumerate(chosen[:14]):
        candidates=[i for i,r in enumerate(rows) if r.get('boxes')]
        anchor=candidates[len(candidates)//2] if candidates else len(rows)//2
        # Include quieter excerpts as well; no claim of random accuracy sampling.
        if index%4==3:
            windows=[i for i in range(0,len(rows)-49,25) if not any(r.get('boxes') for r in rows[i:i+50])]
            if windows:anchor=windows[len(windows)//2]+25
        start=max(0,min(anchor-25,len(rows)-50));clip=render(rows[start:start+50]);clips.append(clip)
        print(json.dumps(dict(completed=len(clips),total=min(14,len(chosen)),session=clip['session'],segment=clip['segment'])),flush=True)
    manifest=ROOT/'artifacts/pipeline/synthetic/a96967e86c9c2652812ac079/manifest.json';doc=json.loads(manifest.read_text());images=[]
    for row in doc['images']:
        source=ROOT/'artifacts'/row['image']
        if sha(source)!=row['sha256']:raise ValueError('Synthetic image checksum mismatch')
        target=MEDIA/(row['id']+'.png');shutil.copyfile(source,target)
        keys=['id','sha256','width','height','capture_time_s','target_present_in_physical_patch','boxes','camera_footprint_m','field_domain_m','implied_nadir_altitude_m','camera_transfer','acceptance_scope','appearance_validation','observable_label_protocol']
        images.append({**{k:row.get(k) for k in keys},'url':'/artifacts/presentation/evidence-browser/'+target.name})
    clips.sort(key=lambda c:-c['positive_frames'])
    catalog=dict(clips=clips,images=images,synthetic_manifest_sha256=sha(manifest),new_inference_calls=0)
    temp=PAGE/'catalog.building.json';temp.write_text(json.dumps(catalog,indent=2));temp.replace(PAGE/'catalog.json')
    shutil.copyfile(pathlib.Path(__file__).with_name('evidence_browser.html'),PAGE/'index.html')
    print(json.dumps(dict(url='https://136.115.179.240/presentation/evidence-browser/index.html',clips=len(clips),images=len(images))))
if __name__=='__main__':main()
