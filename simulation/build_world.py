#!/usr/bin/python3
"""Build an isolated native Gazebo world from the existing BSD3 PX4 x500."""
import copy,hashlib,json,os,shutil,subprocess
from pathlib import Path
import xml.etree.ElementTree as E

SSD=Path(os.environ.get('AEROROZUM_SIM_ROOT','/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/simulation'))
PX4=Path(os.environ.get('AEROROZUM_PX4_SOURCE','/home/yehormsi/src/PX4-Autopilot-v1.17.0'))

def child(parent,tag,text=None,**attrs):
    node=E.SubElement(parent,tag,attrs)
    if text is not None:node.text=str(text)
    return node

def thermal(visual,temperature):
    p=child(visual,'plugin',filename='gz-sim-thermal-system',name='gz::sim::systems::Thermal');child(p,'temperature',temperature)

def camera(link,name,kind,pose,width,height,fov):
    s=child(link,'sensor',name=name,type='thermal' if kind=='thermal' else 'camera');child(s,'pose',pose)
    child(s,'always_on','true');child(s,'update_rate',8);child(s,'topic','/aerorozum_sim/'+name)
    c=child(s,'camera');child(c,'horizontal_fov',fov);i=child(c,'image');child(i,'width',width);child(i,'height',height);child(i,'format','L16' if kind=='thermal' else 'R8G8B8')
    clip=child(c,'clip');child(clip,'near',.05);child(clip,'far',150)
    if kind=='thermal':
        p=child(s,'plugin',filename='gz-sim-thermal-sensor-system',name='gz::sim::systems::ThermalSensor');child(p,'min_temp',270);child(p,'max_temp',330);child(p,'resolution',.01)

def main():
    for name in ['models','worlds','media','logs','cache','config','provenance']:(SSD/name).mkdir(parents=True,exist_ok=True)
    original=PX4/'Tools/simulation/gz/models/x500_base';target=SSD/'models/x500_base'
    if not target.exists():shutil.copytree(original,target)
    shutil.copy2(PX4/'LICENSE',SSD/'provenance/PX4-BSD3-LICENSE')
    root=E.Element('sdf',version='1.9');world=child(root,'world',name='aerorozum_mission')
    physics=child(world,'physics',name='native_multirotor',type='ignored');child(physics,'max_step_size',.004);child(physics,'real_time_factor',1.0)
    child(world,'gravity','0 0 -9.81')
    for file,name in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),('sensors','Sensors')]:
        p=child(world,'plugin',filename='gz-sim-'+file+'-system',name='gz::sim::systems::'+name)
        if name=='Sensors':child(p,'render_engine','ogre2')
    scene=child(world,'scene');child(scene,'ambient','.6 .6 .6 1');child(scene,'background','.35 .5 .65 1')
    light=child(world,'light',name='sun',type='directional');child(light,'pose','0 0 20 0 0 0');child(light,'diffuse','.9 .9 .85 1');child(light,'direction','-.4 .2 -1');child(light,'cast_shadows','true')
    ground=child(world,'model',name='soil');child(ground,'static','true');link=child(ground,'link',name='ground')
    for tag in ['collision','visual']:
        obj=child(link,tag,name='soil_'+tag);plane=child(child(obj,'geometry'),'plane');child(plane,'normal','0 0 1');child(plane,'size','70 70')
        if tag=='visual':
            mat=child(obj,'material');child(mat,'ambient','.32 .38 .2 1');child(mat,'diffuse','.32 .38 .2 1');thermal(obj,295.)
    drone=copy.deepcopy(E.parse(target/'model.sdf').getroot().find('model'));drone.set('name','aerorozum_drone');world.append(drone)
    # Existing x500 motor model and inertial geometry are retained. No PX4,
    # MAVLink, HIL, ROS or real-aircraft plugin is started.
    for plugin in E.parse(PX4/'Tools/simulation/gz/models/x500/model.sdf').getroot().find('model').findall('plugin'):
        p=copy.deepcopy(plugin);child(p,'robotNamespace','aerorozum_sim');child(p,'actuator_number',p.findtext('motorNumber'));drone.append(p)
    sample=E.parse('/usr/share/gz/gz-sim8/worlds/multicopter_velocity_control.sdf').getroot().find('world/include')
    controller=copy.deepcopy(next(p for p in sample.findall('plugin') if p.get('name').endswith('MulticopterVelocityControl')))
    controller.find('robotNamespace').text='aerorozum_sim';controller.find('comLinkName').text='base_link'
    for rotor in controller.findall('rotorConfiguration/rotor'):rotor.find('jointName').text=rotor.findtext('jointName').split('/')[-1]
    drone.append(controller)
    odom=child(drone,'plugin',filename='gz-sim-odometry-publisher-system',name='gz::sim::systems::OdometryPublisher');child(odom,'dimensions',3);child(odom,'odom_publish_frequency',40);child(odom,'odom_topic','/aerorozum_sim/odometry');child(odom,'odom_frame','world');child(odom,'robot_base_frame','base_link')
    base=drone.find("link[@name='base_link']")
    camera(base,'thermal','thermal','0 0 -.12 0 1.57079632679 0',640,512,1.25)
    camera(base,'rgb','rgb','.03 0 -.13 0 1.57079632679 0',640,512,1.25)
    overview=child(world,'model',name='overview_camera');child(overview,'static','true');ol=child(overview,'link',name='camera');camera(ol,'overview','rgb','22 -24 21 0 .57 2.31',960,600,1.32)
    truth=[]
    for index,(x,y,temp,radius) in enumerate([(-10,1,299.2,.15),(-8,-2,292.5,.12),(10,0,298.6,.14),(11,2,296.5,.11),(0,12,299.0,.13),(1,-12,292.8,.16)]):
        model=child(world,'model',name='inert_survey_target_'+str(index));child(model,'static','true');child(model,'pose',f'{x} {y} .045 0 0 0');ln=child(model,'link',name='target')
        for tag in ['visual','collision']:
            v=child(ln,tag,name=tag);g=child(child(v,'geometry'),'cylinder');child(g,'radius',radius);child(g,'length',.09)
            if tag=='visual':
                m=child(v,'material');child(m,'ambient','.25 .22 .15 1');child(m,'diffuse','.25 .22 .15 1');thermal(v,temp)
        truth.append(dict(id=index,x_m=x,y_m=y,temperature_k=temp,radius_m=radius))
    for index,(x,y) in enumerate([(-4,5),(6,8),(14,-6),(-16,-8)]):
        m=child(world,'model',name='vegetation_'+str(index));child(m,'static','true');child(m,'pose',f'{x} {y} .8 0 0 0');ln=child(m,'link',name='bush');v=child(ln,'visual',name='foliage');g=child(child(v,'geometry'),'sphere');child(g,'radius',.8);mat=child(v,'material');child(mat,'ambient','.14 .33 .1 1');child(mat,'diffuse','.14 .33 .1 1');thermal(v,294.5)
    path=SSD/'worlds/aerorozum.sdf';E.indent(root);E.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    (SSD/'provenance/scene.truth.json').write_text(json.dumps({'targets':truth,'thermal_model':'Assigned temperatures for native Gazebo rendering; no coupled soil heat/moisture validation'},indent=2))
    provenance={'px4_source':str(PX4),'commit':subprocess.check_output(['git','-C',str(PX4),'rev-parse','HEAD'],text=True).strip(),'model_sdf_sha256':hashlib.sha256((original/'model.sdf').read_bytes()).hexdigest(),'world_sdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dynamics':'Native Gazebo physics, existing x500 motor models, native MulticopterVelocityControl','real_drone_connections':False}
    (SSD/'provenance/world.json').write_text(json.dumps(provenance,indent=2));print(path)

if __name__=='__main__':main()
