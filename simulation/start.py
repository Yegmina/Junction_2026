#!/usr/bin/python3
"""Install/start persistent user services; all simulation storage stays on SSD."""
import argparse,json,os,shlex,subprocess,time,urllib.request
from pathlib import Path
from build_world import SSD

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replace-transient',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if not (SSD/'worlds/aerorozum.sdf').is_file():raise RuntimeError('Build world on mounted SSD first')
    units=SSD/'units';units.mkdir(exist_ok=True);config=Path.home()/'.config/systemd/user';config.mkdir(parents=True,exist_ok=True)
    env={'GZ_PARTITION':'aerorozum-mission-local','GZ_IP':'127.0.0.1','GZ_DISCOVERY_MSG_PORT':'18935','GZ_DISCOVERY_SRV_PORT':'18936','GZ_SIM_RESOURCE_PATH':str(SSD/'models'),'GZ_FUEL_CACHE_PATH':str(SSD/'fuel'),'GZ_SIM_LOG_ROOT':str(SSD/'logs'),'GZ_LOG_PATH':str(SSD/'logs'),'XDG_CACHE_HOME':str(SSD/'cache'),'XDG_CONFIG_HOME':str(SSD/'config'),'__GL_SHADER_DISK_CACHE_PATH':str(SSD/'cache/nvidia'),'MESA_SHADER_CACHE_DIR':str(SSD/'cache/mesa')}
    prefix='[Unit]\nDescription=AeroRozum local native Gazebo\n'
    common='[Service]\nType=simple\nRestart=on-failure\nRestartSec=3\nWorkingDirectory='+str(SSD/'runtime')+'\n'
    environment=''.join('Environment='+json.dumps(k+'='+v)+'\n' for k,v in env.items())
    server=prefix+common+environment+'ExecStart=/usr/bin/gz sim -s -r --headless-rendering -v 2 '+json.dumps(str(SSD/'worlds/aerorozum.sdf'))+'\nMemoryMax=6G\nStandardOutput=append:'+str(SSD/'logs/server.log')+'\nStandardError=append:'+str(SSD/'logs/server.log')+'\n[Install]\nWantedBy=default.target\n'
    adapter=prefix+'Requires=aerorozum-gazebo-server.service\nAfter=aerorozum-gazebo-server.service\nPartOf=aerorozum-gazebo-server.service\n'+common+environment+'ExecStart=/usr/bin/python3 '+json.dumps(str(repo/'simulation/runtime.py'))+'\nMemoryMax=2G\nStandardOutput=append:'+str(SSD/'logs/adapter.log')+'\nStandardError=append:'+str(SSD/'logs/adapter.log')+'\n[Install]\nWantedBy=default.target\n'
    names=['aerorozum-gazebo-server.service','aerorozum-gazebo-adapter.service']
    if a.replace_transient:subprocess.run(['systemctl','--user','stop',*reversed(names)],check=False)
    for name,body in zip(names,[server,adapter]):
        path=units/name;path.write_text(body);link=config/name
        if link.is_symlink():link.unlink()
        elif link.exists():raise RuntimeError('Existing unrelated unit needs review: '+str(link))
        link.symlink_to(path)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','reset-failed',*names],check=False,capture_output=True)
    subprocess.run(['systemctl','--user','enable',*names],check=True)
    subprocess.run(['systemctl','--user','start',names[1]],check=True)
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        try:
            state=json.load(urllib.request.urlopen('http://127.0.0.1:18930/status',timeout=2))
            if state.get('ready') and 'thermal' in state.get('frames',{}):print(json.dumps({'ready':True,'pose':state['pose'],'cameras':list(state['frames']),'persistent_user_units':names}));return
        except OSError:pass
        time.sleep(.25)
    raise RuntimeError('Native Gazebo startup timeout; inspect SSD logs')

if __name__=='__main__':main()
