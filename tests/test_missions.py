import asyncio
import pytest
from PIL import Image
from aerorozum.missions import MissionManager,parse_mission_request

class Bridge:
    """Test double only: production has no fallback for missing Gazebo."""
    def __init__(self):self.p=[0,0,.15];self.t=0;self.commands=[];self.paused=False
    def status(self):return dict(ready=True,pose=dict(zip(('x_m','y_m','z_m'),self.p)),velocity=dict(x_m_s=0,y_m_s=0,z_m_s=0),sim_time_s=self.t)
    def resume(self):self.paused=False
    def pause(self):self.paused=True
    def tick(self,seconds):self.t+=seconds
    def set_target(self,x,y,z,yaw_rad=0):self.p=[x,y,z];self.commands.append(list(self.p))
    def hover(self):self.set_target(*self.p)
    def takeoff(self,z):self.set_target(0,0,z)
    def land(self):self.set_target(0,0,.15)
    def capture(self,path,kind):
        import hashlib
        path.parent.mkdir(parents=True,exist_ok=True);Image.new('L',(64,48),120).save(path)
        return dict(camera_topic='/test/thermal',pose=self.status()['pose'],sim_time_s=self.t,atomic_capture=True,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def test_request_constraints_and_budget():
    p=parse_mission_request('Survey north and south, avoid west; revisit uncertain candidates in 2 minutes at 4m')
    assert p['allowed_areas']==['north','south'] and p['forbidden_areas']==['west']
    assert p['altitude_m']==4 and p['budget_s']==120 and p['revisit_uncertain']
    with pytest.raises(ValueError):parse_mission_request('avoid east avoid west avoid north avoid south')

def test_reference_uses_native_capture_and_returns(tmp_path):
    async def run():
        bridge=Bridge();manager=MissionManager(tmp_path,bridge=bridge,detector=lambda _:dict(boxes=[],uncertain_boxes=[],model='test'))
        state=await manager.create('Survey east in 2 minutes','reference')
        await manager.tasks[state['id']];result=manager.get(state['id'])
        assert result['status']=='completed',result.get('error')
        assert len(result['observations'])==1 and result['observations'][0]['pose']['x_m']==10
        assert result['report']['returned_home']
        assert result['report']['coverage']['all_requested_areas_inspected']
        assert result['observations'][0]['source_kind']=='gazebo_camera'
        assert not any(k.startswith('_') for k in result)
    asyncio.run(run())

def test_agent_invalid_area_is_rejected_and_landed(tmp_path):
    class Provider:
        async def generate(self,*args,**kwargs):return dict(action='inspect_area',area_id='west',altitude_m=5)
    async def run():
        bridge=Bridge();manager=MissionManager(tmp_path,bridge=bridge,provider_factory=Provider)
        state=await manager.create('Survey east only','agent');await manager.tasks[state['id']]
        result=manager.get(state['id'])
        assert result['status']=='failed' and 'constraints' in result['error']
        assert result['report']['returned_home']
        assert all(o['area_id']!='west' for o in result['observations'])
        assert all(command[0]>=0 for command in bridge.commands)
        assert not result['report']['coverage']['all_requested_areas_inspected']
        assert not bridge.paused and manager.active is None
    asyncio.run(run())

def test_unavailable_simulator_never_creates_mission(tmp_path):
    class Missing:
        def status(self):return dict(ready=False)
    async def run():
        manager=MissionManager(tmp_path,bridge=Missing())
        with pytest.raises(ValueError,match='not ready'):await manager.create('Survey east','reference')
        assert manager.list()==[]
    asyncio.run(run())

def test_concurrent_launches_cannot_control_one_drone(tmp_path):
    async def run():
        manager=MissionManager(tmp_path,bridge=Bridge(),detector=lambda _:dict(boxes=[],model='test'))
        results=await asyncio.gather(manager.create('Survey east','reference'),manager.create('Survey north','reference'),return_exceptions=True)
        assert sum(isinstance(v,ValueError) for v in results)==1
        await asyncio.gather(*manager.tasks.values())
    asyncio.run(run())

def test_passing_through_target_is_not_arrival(tmp_path):
    class Moving(Bridge):
        def __init__(self):super().__init__();self.p=[0,0,3];self.samples=0
        def status(self):
            self.samples+=1;s=super().status()
            if self.samples<=4:s['velocity']['z_m_s']=1.4
            return s
    async def run():
        b=Moving();m=MissionManager(tmp_path,bridge=b)
        state=dict(id='test',pose=None,route=[],budget=dict(limit_s=120,elapsed_s=0,remaining_s=120),_start_sim=None,_inference_s=0,_stop=False)
        assert await m._wait(state,[0,0,3])
        assert b.samples>=7 and b.t>=1.2
    asyncio.run(run())

def test_lower_revisit_stays_in_the_inspection_area(tmp_path):
    async def run():
        b=Bridge();b.p=[10,0,3];m=MissionManager(tmp_path,bridge=b)
        s=dict(id='revisit',pose=dict(x_m=10,y_m=0,altitude_m=3),plan={},route=[],budget=dict(limit_s=120,elapsed_s=0,remaining_s=120),_start_sim=None,_inference_s=0,_stop=False)
        assert await m._move(s,10,0,2)
        assert b.commands==[[10,0,2]]
    asyncio.run(run())

def test_paired_physics_and_rgb_captures_are_saved_and_sent_to_agent(tmp_path):
    class Paired(Bridge):
        def capture_pair(self,folder,stem):
            thermal=self.capture(folder/(stem+'.png'),'thermal');rgb=self.capture(folder/(stem+'-rgb.png'),'rgb')
            for receipt,path in ((thermal,folder/(stem+'.png')),(rgb,folder/(stem+'-rgb.png'))):receipt.update(path=str(path),pair_id='sensor-pair')
            thermal.update(source_kind='dumux_field_crop',field_sha256='a'*64)
            return dict(thermal=thermal,rgb=rgb)
    class Provider:
        def __init__(self):self.counts=[]
        async def generate(self,*args,**kwargs):
            self.counts.append(len(kwargs['images']))
            return dict(action='inspect_area',area_id='east',altitude_m=3) if len(self.counts)==1 else dict(action='return_home',detections=[])
    async def run():
        provider=Provider();m=MissionManager(tmp_path,bridge=Paired(),provider_factory=lambda:provider,detector=lambda _:dict(boxes=[],model='local_thermal_contrast'))
        s=await m.create('Inspect east at 3m','agent');await m.tasks[s['id']];s=m.get(s['id'])
        assert s['status']=='completed',s['error']
        assert provider.counts==[2,2]
        row=next(o for o in s['observations'] if o['area_id']=='east');assert row['paired_capture_id']=='sensor-pair' and (tmp_path/row['rgb_image']).is_file()
        assert row['evidence']['context']['source_kind']=='dumux_field_crop'
    asyncio.run(run())
