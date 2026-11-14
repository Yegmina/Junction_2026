import {useEffect,useRef,useState} from 'react';
import type {PointerEvent as ReactPointerEvent} from 'react';
import {Crosshair,Expand,Focus,Layers,Minus,Plus,RotateCcw,TriangleAlert,Video} from 'lucide-react';

type Data=Record<string,any>;
type View={mode:string;distance_m:number;elevation_deg:number;azimuth_deg:number;pan_x_m:number;pan_y_m:number};
const follow:View={mode:'follow',distance_m:7,elevation_deg:25,azimuth_deg:-48,pan_x_m:0,pan_y_m:0};
const presets:Record<string,View>={follow,wide:{...follow,mode:'overview',distance_m:48,elevation_deg:48},top:{...follow,distance_m:15,elevation_deg:84,azimuth_deg:-90}};
const clamp=(n:number,min:number,max:number)=>Math.max(min,Math.min(max,n));
const media=(path:string)=>`/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`;
const active=(status:string)=>['planning','starting','running','returning','stopping'].includes(status);

export default function NativeScene({mission,simulation}:{mission:Data|null;simulation:Data}){
 const[frame,setFrame]=useState(''),[error,setError]=useState(''),[viewError,setViewError]=useState(''),[captured,setCaptured]=useState('');
 const[live,setLive]=useState(true),[view,setView]=useState<View>(follow),[imageView,setImageView]=useState({scale:1,x:0,y:0});
 const stage=useRef<HTMLDivElement>(null),panel=useRef<HTMLElement>(null),objectUrl=useRef(''),current=useRef(view),pending=useRef<View|null>(null),interacted=useRef(false);
 const saved=(Array.isArray(mission?.observations)?mission!.observations:[]).slice().reverse().find((o:Data)=>typeof o.overview_image==='string');
 const enabled=simulation.available===true&&(live||!saved),savedMode=!!saved&&!enabled,imagePath=savedMode?media(saved.overview_image):frame;
 const state=useRef({savedMode,enabled,imageView});state.current={savedMode,enabled,imageView};current.current=view;
 const pointers=useRef(new Map<number,{x:number;y:number}>()),drag=useRef<{x:number;y:number;pan:boolean}|null>(null),pinch=useRef(0);
 const update=(change:Partial<View>)=>{interacted.current=true;const next={...current.current,...change};current.current=next;setView(next);pending.current=next;setViewError('')};
 const preset=(name:string)=>{setLive(true);update(presets[name])};
 const zoom=(delta:number)=>{
  if(state.current.savedMode)setImageView(v=>({...v,scale:clamp(v.scale*Math.exp(-delta*.0015),1,8)}));
  else if(state.current.enabled)update({distance_m:clamp(current.current.distance_m*Math.exp(delta*.0015),3,85)});
 };
 const zoomRef=useRef(zoom);zoomRef.current=zoom;
 useEffect(()=>{
  setLive(!saved||active(mission?.status));setImageView({scale:1,x:0,y:0});setError('');
 },[mission?.id]);
 useEffect(()=>{
  const node=stage.current;if(!node)return;
  const wheel=(e:WheelEvent)=>{if(!state.current.enabled&&!state.current.savedMode)return;e.preventDefault();zoomRef.current(clamp(e.deltaY,-180,180))};
  node.addEventListener('wheel',wheel,{passive:false});return()=>node.removeEventListener('wheel',wheel);
 },[]);
 useEffect(()=>{
  let alive=true,busy=false;const abort=new AbortController();
  fetch('/api/missions/view',{signal:abort.signal}).then(r=>r.ok?r.json():null).then(d=>{if(alive&&d?.settings&&!interacted.current){current.current=d.settings;setView(d.settings)}}).catch(()=>{});
  const timer=setInterval(async()=>{if(busy||!pending.current)return;const value=pending.current;pending.current=null;busy=true;
   try{const response=await fetch('/api/missions/view',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value),signal:abort.signal});if(!response.ok){const d=await response.json();throw new Error(d.detail||'The overview camera is unavailable.')}}
   catch(e){if(alive&&(e as Error).name!=='AbortError')setViewError((e as Error).message)}finally{busy=false}
  },150);
  return()=>{alive=false;abort.abort();clearInterval(timer)};
 },[]);
 useEffect(()=>{
  let alive=true,busy=false;const abort=new AbortController();
  const capture=async()=>{if(busy||!enabled)return;busy=true;try{const response=await fetch(`/api/missions/camera/overview?t=${Date.now()}`,{signal:abort.signal,cache:'no-store'});if(!response.ok)throw new Error('The overview camera is waiting for an image.');const blob=await response.blob();if(!blob.type.startsWith('image/'))throw new Error('Camera image unavailable.');if(alive){const next=URL.createObjectURL(blob);if(objectUrl.current)URL.revokeObjectURL(objectUrl.current);objectUrl.current=next;setFrame(next);setCaptured(new Date().toLocaleTimeString('en-GB'));setError('')}}catch(e){if(alive&&(e as Error).name!=='AbortError')setError((e as Error).message)}finally{busy=false}};
  capture();const timer=enabled?setInterval(capture,700):undefined;return()=>{alive=false;abort.abort();if(timer)clearInterval(timer)};
 },[enabled]);
 useEffect(()=>()=>{if(objectUrl.current)URL.revokeObjectURL(objectUrl.current)},[]);
 const down=(e:ReactPointerEvent<HTMLDivElement>)=>{if(!imagePath)return;e.currentTarget.setPointerCapture(e.pointerId);pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});drag.current={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};if(pointers.current.size===2){const[a,b]=[...pointers.current.values()];pinch.current=Math.hypot(a.x-b.x,a.y-b.y)}};
 const move=(e:ReactPointerEvent<HTMLDivElement>)=>{
  if(!pointers.current.has(e.pointerId))return;pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});
  if(pointers.current.size===2){const[a,b]=[...pointers.current.values()],distance=Math.hypot(a.x-b.x,a.y-b.y);if(pinch.current>0)zoom(-Math.log(Math.max(1,distance)/pinch.current)/.0015);pinch.current=distance;return}
  const previous=drag.current;if(!previous)return;const dx=e.clientX-previous.x,dy=e.clientY-previous.y;drag.current={...previous,x:e.clientX,y:e.clientY};
  if(savedMode){setImageView(v=>({...v,x:clamp(v.x+dx,-400*(v.scale-1),400*(v.scale-1)),y:clamp(v.y+dy,-250*(v.scale-1),250*(v.scale-1))}));return}
  if(!enabled)return;
  if(previous.pan||e.shiftKey){const az=current.current.azimuth_deg*Math.PI/180,k=current.current.distance_m/500;update({pan_x_m:clamp(current.current.pan_x_m+(dx*Math.sin(az)-dy*Math.cos(az))*k,-50,50),pan_y_m:clamp(current.current.pan_y_m+(-dx*Math.cos(az)-dy*Math.sin(az))*k,-50,50)})}
  else update({azimuth_deg:((current.current.azimuth_deg+dx*.35+180)%360+360)%360-180,elevation_deg:clamp(current.current.elevation_deg+dy*.2,8,88)});
 };
 const up=(e:ReactPointerEvent<HTMLDivElement>)=>{pointers.current.delete(e.pointerId);pinch.current=0;const left=[...pointers.current.values()][0];drag.current=left?{...left,pan:false}:null};
 return <section className="ms-native panel" ref={panel}>
  <div className="panel-top"><div className="panel-title"><Video size={17}/><h2>Native Gazebo scene</h2><span className="ms-map-note">INTERACTIVE CAMERA</span></div><span className={`ms-native-live ${enabled&&frame&&!error?'running':''}`}><i/>{savedMode?'Saved mission frame':enabled?frame?'Live camera':'Connecting camera':'Camera unavailable'}</span></div>
  <div className="ms-native-toolbar"><div className="ms-native-presets"><button className={enabled&&view.mode==='follow'&&view.elevation_deg<80?'selected':''} disabled={!simulation.available} onClick={()=>preset('follow')}><Focus size={13}/>Follow drone</button><button className={enabled&&view.mode==='overview'?'selected':''} disabled={!simulation.available} onClick={()=>preset('wide')}><Layers size={13}/>Wide</button><button className={enabled&&view.elevation_deg>=80?'selected':''} disabled={!simulation.available} onClick={()=>preset('top')}><Crosshair size={13}/>Top</button></div><div className="ms-native-utility">{saved&&<button onClick={()=>setLive(!enabled)}>{savedMode?'Live camera':'Saved frame'}</button>}<button aria-label="Expand scene" title="Expand scene" onClick={()=>panel.current?.requestFullscreen?.().catch(()=>{})}><Expand size={14}/></button></div></div>
  <div className={`ms-native-stage ${imagePath?'interactive':''}`} ref={stage} tabIndex={0} aria-label="Gazebo camera: drag to orbit, Shift-drag to pan, scroll to zoom" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onContextMenu={e=>e.preventDefault()}>
   {imagePath?<img src={imagePath} draggable={false} style={savedMode?{transform:`translate(${imageView.x}px,${imageView.y}px) scale(${imageView.scale})`}:undefined} alt={savedMode?'Saved Gazebo mission capture':'Live native Gazebo camera following the drone'}/>:<div className="ms-camera-empty"><Video size={34}/><strong>{enabled?'Connecting to the scene':'Gazebo camera'}</strong><p>{enabled?'The live scene will appear here.':'The native simulator is reconnecting.'}</p></div>}
   {imagePath&&<><div className="ms-native-view-label"><span>{savedMode?'SAVED IMAGE':view.mode==='follow'?'FOLLOWING DRONE':'WORLD VIEW'}</span><strong>{savedMode?`${imageView.scale.toFixed(1)}×`:view.distance_m.toFixed(1)+' m'}</strong></div><div className="ms-native-zoom"><button aria-label="Zoom in" title="Zoom in" onPointerDown={e=>e.stopPropagation()} onClick={()=>zoom(-160)}><Plus size={16}/></button><button aria-label="Zoom out" title="Zoom out" onPointerDown={e=>e.stopPropagation()} onClick={()=>zoom(160)}><Minus size={16}/></button><i/><button aria-label="Reset camera view" title="Reset view" onPointerDown={e=>e.stopPropagation()} onClick={()=>savedMode?setImageView({scale:1,x:0,y:0}):preset('follow')}><RotateCcw size={14}/></button></div></>}
   {(error||viewError)&&enabled&&<div className="ms-native-error"><TriangleAlert size={13}/>{error||viewError}</div>}
  </div>
  <footer><span>{savedMode?'Saved native camera capture':`Native Gazebo camera${captured?' · '+captured:''}`}</span><span>{savedMode?'Drag to pan · Scroll or pinch to zoom':'Drag to orbit · Shift / right-drag to pan · Scroll or pinch to zoom'}</span></footer>
 </section>
}
