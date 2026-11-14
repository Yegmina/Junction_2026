import asyncio
import copy
import math
import pytest
from PIL import Image
from aerorozum.missions import MissionManager,BOUNDS,AREAS
from aerorozum.mission_navigation import validate_target,return_route,intersects


class MovingBridge:
    """Deterministic test motion, not the native simulator used in production."""
    def __init__(self):
        self.p=[0.,0.,.15];self.target=list(self.p);self.t=0.;self.commands=[];self.pause_calls=0
    def status(self):
        prior=list(self.p);self.p=[p+max(-.3,min(.3,t-p)) for p,t in zip(self.p,self.target)];self.t+=.1
        return dict(ready=True,pose=dict(zip(('x_m','y_m','z_m'),self.p)),velocity=dict(zip(('x_m_s','y_m_s','z_m_s'),[(p-a)/.1 for p,a in zip(self.p,prior)])),sim_time_s=self.t)
    def resume(self):pass
    def pause(self):self.pause_calls+=1
    def tick(self,seconds):self.t+=seconds
    def set_target(self,x,y,z,yaw_rad=0):self.target=[x,y,z];self.commands.append(list(self.target))
    def hover(self):self.target=list(self.p)
    def takeoff(self,z):self.set_target(0,0,z)
    def land(self):self.set_target(0,0,.15)
    def capture(self,path,kind):
        import hashlib
        path.parent.mkdir(parents=True,exist_ok=True);Image.new('L',(64,48),120).save(path)
        return dict(camera_topic='/test/'+kind,pose=dict(zip(('x_m','y_m','z_m'),self.p)),sim_time_s=self.t,atomic_capture=True,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def detector(_):return dict(boxes=[],uncertain_boxes=[],model='local_thermal_contrast')


def test_free_goto_retargets_while_aircraft_and_telemetry_keep_moving(tmp_path):
    async def run():
        bridge=MovingBridge()
        class Provider:
            def __init__(self):self.calls=[];self.changed_during_review=[]
            async def generate(self,kind,prompt,**kwargs):
                before=list(bridge.p);await asyncio.sleep(.25);self.changed_during_review.append(before!=bridge.p);self.calls.append(prompt)
                i=len(self.calls)
                if i==3:return dict(tool='return_home',rationale='Evidence collected; return now.')
                return dict(tool='goto',arguments=dict(x_m=18 if i==1 else 5,y_m=12 if i==1 else -7,altitude_m=4 if i==1 else 3),review_after_s=.5,observation_summary='Reviewing the latest thermal frame.',rationale='Adjust course based on new evidence.',next_check='Review the next camera frame.')
        provider=Provider();manager=MissionManager(tmp_path,bridge=bridge,provider_factory=lambda:provider,detector=detector)
        m=await manager.create('Inspect custom coordinates at 4m in 3 minutes','agent');await manager.tasks[m['id']];s=manager.get(m['id'])
        assert s['status']=='completed',s['error']
        assert [w['x_m'] for w in s['plan']['waypoints']]==[18.,5.]
        assert s['agent_steps'][1]['adjusted_during_motion'] is True
        assert any(provider.changed_during_review[1:]) and bridge.pause_calls==0
        assert s['agent_steps'][0]['result']['status']=='moving'
        assert s['agent_steps'][1]['rationale']=='Adjust course based on new evidence.'
        assert s['report']['returned_home'] and len(s['observations'])>=3
    asyncio.run(run())


def test_invalid_coordinates_are_visible_tool_feedback_and_can_be_corrected(tmp_path):
    async def run():
        bridge=MovingBridge()
        class Provider:
            def __init__(self):self.prompts=[]
            async def generate(self,kind,prompt,**kwargs):
                self.prompts.append(prompt)
                if len(self.prompts)==3:return dict(tool='return_home')
                return dict(tool='goto',arguments=dict(x_m=999 if len(self.prompts)==1 else 4,y_m=1,altitude_m=3),review_after_s=.5)
        provider=Provider();manager=MissionManager(tmp_path,bridge=bridge,provider_factory=lambda:provider,detector=detector)
        m=await manager.create('Inspect at 3m in 3 minutes','agent');await manager.tasks[m['id']];s=manager.get(m['id'])
        assert s['status']=='completed',s['error']
        assert s['agent_steps'][0]['status']=='rejected'
        assert 'geofence' in provider.prompts[1] and all(abs(c[0])<=24 for c in bridge.commands)
        assert any(c[:2]==[4.,1.] for c in bridge.commands)
    asyncio.run(run())


def test_stop_cancels_pending_inference_and_returns_home(tmp_path):
    async def run():
        bridge=MovingBridge();started=asyncio.Event();cancelled=asyncio.Event()
        class Provider:
            async def generate(self,*args,**kwargs):
                started.set()
                try:await asyncio.Event().wait()
                finally:cancelled.set()
        manager=MissionManager(tmp_path,bridge=bridge,provider_factory=Provider,detector=detector)
        m=await manager.create('Survey at 3m in 3 minutes','agent');await asyncio.wait_for(started.wait(),5)
        await manager.stop(m['id']);await asyncio.wait_for(manager.tasks[m['id']],8);s=manager.get(m['id'])
        assert cancelled.is_set() and s['status']=='stopped' and s['report']['returned_home']
        assert bridge.pause_calls==0
    asyncio.run(run())


def test_exclusions_apply_to_the_whole_goto_segment_and_return_can_route_around():
    west=next(a['bounds'] for a in AREAS if a['id']=='west')
    with pytest.raises(ValueError,match='excluded'):
        validate_target(-20.,0.,4.,BOUNDS,[west],start=(0.,0.))
    route=return_route((-20.,0.),(0.,0.),BOUNDS,[west]);assert len(route)>1
    assert not any(intersects(a,b,west) for a,b in zip([(-20.,0.)]+route,route))


@pytest.mark.parametrize('point',[(float('nan'),0,3),(0,0,True),(25,0,3),(0,0,15)])
def test_invalid_goto_values_are_rejected(point):
    with pytest.raises(ValueError):validate_target(*point,BOUNDS)
