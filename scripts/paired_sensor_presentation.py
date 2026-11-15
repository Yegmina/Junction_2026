"""Presentation comparison of native RGB and fixed-pose physics snapshots."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from aerorozum.common import data_root,save
from simulation.physics_camera import PhysicsCamera

def main():
    root=data_root();mission='ca1f8d19b1b64ab3';s=json.loads((root/'missions'/mission/'state.json').read_text());o=s['observations'][0]
    scene=root.parent/'simulation/sensor-scene';cold=PhysicsCamera(scene,'cooling-scene.json');cooling,meta=cold.render(o['capture']['pose'],sim_time_s=o['capture']['sim_time_s'])
    output=root/'presentation';output.mkdir(exist_ok=True);font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    f=lambda size:ImageFont.truetype(font,size)
    canvas=Image.new('RGB',(1920,710),'#10231e');d=ImageDraw.Draw(canvas)
    d.text((24,16),'AEROROZUM  /  PAIRED RGB + THERMAL PHYSICS',font=f(26),fill='#d8e9d1')
    panels=[('Native RGB · visible light',Image.open(root/o['rgb_image'])),('Warming · DuMux time 7.5 h',Image.open(root/o['image'])),('Cooling · DuMux time 17 h',cooling)]
    for i,(title,image) in enumerate(panels):
        d.text((i*640+18,65),title,font=f(21),fill='#b9dba9');canvas.paste(image.convert('RGB').resize((640,512)),(i*640,108))
    d.text((24,636),'Same camera pose. RGB + warming are captured together; cooling is re-rendered at that exact pose.',font=f(18),fill='#b3c6ba')
    d.text((24,670),'Experimental DuMux composition + reduced-order ground response. Numerical/field validation pending.',font=f(17),fill='#92aea0')
    path=output/'paired-physics-comparison.png';canvas.save(path)
    save(output/'paired-physics-comparison.json',dict(mission_id=mission,observation_id=o['id'],native_rgb_capture=o['rgb_capture'],warming_capture=o['capture'],cooling_render=dict(**meta,pose=o['capture']['pose'],comparison_only=True),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(path)

if __name__=='__main__':main()
