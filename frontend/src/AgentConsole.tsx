import {ArrowRight,CheckCircle2,Code2,LoaderCircle,Navigation,Radar,TriangleAlert} from 'lucide-react';

type Data=Record<string,any>;
const number=(value:any)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1):'—';
const label=(value:any)=>String(value||'').replaceAll('_',' ');
function command(step:Data){
 const a=step.arguments||{};
 if(step.tool==='goto')return `goto(x=${number(a.x_m)}, y=${number(a.y_m)}, altitude=${number(a.altitude_m)}${typeof a.yaw_deg==='number'?`, yaw=${number(a.yaw_deg)}°`:''})`;
 return step.tool?`${step.tool}()`:'Reviewing camera evidence and flight feedback…';
}

export default function AgentConsole({mission}:{mission:Data|null}){
 const steps:Data[]=Array.isArray(mission?.agent_steps)?mission!.agent_steps:[];
 const phase=mission?.agent_phase||mission?.status||'waiting',pose=mission?.pose,nav=mission?.navigation||{};
 const busy=phase==='deciding',items=steps.slice(-16).reverse();
 return <section className="ms-agent-console panel">
  <div className="panel-top"><div className="panel-title"><Code2 size={17}/><h2>Agent activity</h2></div><span className={`ms-agent-phase ${busy?'thinking':''}`}>{busy?<LoaderCircle size={12} className="spin"/>:<Radar size={12}/>} {label(phase)}</span></div>
  <div className="ms-agent-telemetry"><span>E <b>{number(pose?.x_m)}</b> m</span><span>N <b>{number(pose?.y_m)}</b> m</span><span>Height <b>{number(pose?.altitude_m)}</b> m</span>{nav.target&&<small><Navigation size={11}/> {label(nav.status)} · {number(nav.distance_m)} m to target</small>}</div>
  <div className="ms-agent-steps" aria-live="polite" aria-label="Agent decisions and tool feedback">
   {items.length?items.map((step,i)=><article key={step.id||i} className={`ms-agent-step ${step.status}`}>
    <header><span>TURN {String(steps.length-i).padStart(2,'0')}</span><span>{step.status==='deciding'?<LoaderCircle size={12} className="spin"/>:step.status==='rejected'?<TriangleAlert size={12}/>:<CheckCircle2 size={12}/>} {label(step.status)}</span></header>
    {step.observation_summary&&<div className="ms-agent-note"><small>Observed</small><p>{step.observation_summary}</p></div>}
    {step.rationale&&<div className="ms-agent-note"><small>{step.source==='controller_guard'?'Controller guard':'Decision rationale'}</small><p>{step.rationale}</p></div>}
    <pre className="ms-agent-command"><Code2 size={13}/><code>{command(step)}</code></pre>
    {step.adjusted_during_motion&&<span className="ms-agent-adjustment"><ArrowRight size={11}/>Adjusted destination during flight</span>}
    {step.next_check&&<div className="ms-agent-note next"><small>Next check</small><p>{step.next_check}</p></div>}
    {step.result&&<div className="ms-agent-result"><CheckCircle2 size={12}/><span>{label(step.result.status)}{step.result.pose&&` · E ${number(step.result.pose.x_m)}, N ${number(step.result.pose.y_m)}, h ${number(step.result.pose.altitude_m)} m`}{step.result.navigation?.distance_m!==undefined&&` · ${number(step.result.navigation.distance_m)} m remaining`}</span></div>}
    {step.error&&<div className="ms-agent-rejection"><TriangleAlert size={12}/>{step.error}</div>}
   </article>):<div className="ms-agent-empty"><Code2 size={28}/><strong>Watch the agent work</strong><p>Observations, decision summaries, coordinate commands and flight feedback appear here.</p></div>}
  </div>
  <footer>Local coordinates in metres · flight continues during AI review</footer>
 </section>
}
