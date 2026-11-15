#!/usr/bin/python3
"""Prepare recorded thermal pixels as a native world heat-signature material."""
import hashlib,json,shutil
import xml.etree.ElementTree as E
from PIL import Image
from build_world import SSD,child

def main():
    artifacts=SSD.parent/'artifacts';doc=json.loads((artifacts/'pipeline/inputs/backgrounds.json').read_text())
    rows=[r for r in doc['crops'] if r.get('split')=='train' and r.get('row_kind')=='zip_screened_background' and r.get('screening',{}).get('status')=='accepted' and r.get('detections')==[] and 15<r.get('image_statistics',{}).get('gray_stddev',0)<65]
    row=max(rows,key=lambda r:min(r['image_statistics']['width'],r['image_statistics']['height']));source=artifacts/row['image']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Crop hash differs')
    model=SSD/'models/recorded_soil';texture=model/'materials/textures/recorded_ground.png';texture.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,texture)
    root=E.Element('sdf',version='1.9');m=child(root,'model',name='recorded_soil_ground');child(m,'static','true');link=child(m,'link',name='ground')
    for tag in ['collision','visual']:
        obj=child(link,tag,name='recorded_'+tag);p=child(child(obj,'geometry'),'plane');child(p,'normal','0 0 1');child(p,'size','70 70')
        if tag=='visual':
            mat=child(obj,'material');child(mat,'ambient','1 1 1 1');child(mat,'diffuse','1 1 1 1');metal=child(child(mat,'pbr'),'metal');child(metal,'albedo_map',texture)
            plugin=child(obj,'plugin',filename='gz-sim-thermal-system',name='gz::sim::systems::Thermal');child(plugin,'heat_signature',texture);child(plugin,'min_temp',293.);child(plugin,'max_temp',297.)
    E.indent(root);E.ElementTree(root).write(model/'model.sdf',encoding='unicode',xml_declaration=True)
    proof={'source_path':str(source),'source_sha256':row['sha256'],'source_archive':row['source'],'screening':row['screening'],'target_free_verified':False,'radiometric':False,'temperature_mapping':{'minimum_k':293.,'maximum_k':297.,'assumed_not_measured':True},'world_extent_m':[70,70],'reprojection':'Native Gazebo UV rendering from moving camera; no replay','source_width_height':Image.open(source).size,'model_path':str(model/'model.sdf')}
    (SSD/'provenance/recorded-ground.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
