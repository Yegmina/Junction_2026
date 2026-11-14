import {Sparkles} from 'lucide-react';
type D=Record<string,any>;
function Curve({rows,field}:{rows:D[];field:string}){
 const points=rows.filter(r=>typeof r[field]==='number'&&typeof r.step==='number');
 if(points.length<2)return <div className="pl-learning-wait">Waiting for measured {field} logs</div>;
 const lo=Math.min(0,...points.map(r=>r[field])),hi=Math.max(...points.map(r=>r[field])),maxStep=Math.max(1,...points.map(r=>r.step));
 const seeds=[...new Set(points.map(r=>r.seed??17))];
 return <svg viewBox="0 0 540 160" className="pl-learning-svg" role="img" aria-label={`Measured training ${field}`}>
 {[30,80,130].map(y=><line key={y} x1="40" x2="525" y1={y} y2={y} stroke="#2b3e34" strokeDasharray="3 5"/>)}
 {seeds.map((seed,i)=><polyline key={seed} points={points.filter(r=>(r.seed??17)===seed).map(r=>`${40+r.step/maxStep*480},${130-(r[field]-lo)/Math.max(1e-9,hi-lo)*100}`).join(' ')} fill="none" stroke={i===0?'#c4e99e':'#8abcb8'} strokeWidth="2" strokeDasharray={i===0?undefined:'5 3'}/>)}
 <text x="2" y="34" fill="#819b87" fontSize="9">{hi.toFixed(3)}</text><text x="2" y="133" fill="#819b87" fontSize="9">{lo.toFixed(3)}</text><text x="40" y="153" fill="#819b87" fontSize="9">Optimizer update</text><text x="499" y="153" fill="#819b87" fontSize="9">{maxStep}</text>
 </svg>
}
export default function LearningSignals({training}:{training:D}){
 const metrics=training.metrics||[],active=training.active||[];
 return <section className="panel pl-learning"><div className="panel-top"><div className="panel-title"><Sparkles size={17}/><h2>Learning signals</h2></div><span className="mono tiny">ACTUAL OPTIMIZER LOGS</span></div><div className="pl-learning-columns"><div><h3>Training loss <span>lower is a training objective</span></h3><Curve rows={metrics} field="loss"/></div><div><h3>Detection reward <span>training signal, not held-out accuracy</span></h3><Curve rows={metrics} field="reward"/></div></div><div className="pl-learning-seeds">{active.map((a:D)=><span key={a.job_id}><i/>Seed {a.seed} · GPU {(a.gpu_group||[]).join(' + ')} · {String(a.stage||'starting').replaceAll('_',' ')}{a.evaluation_total?` · ${a.evaluation_completed||0} / ${a.evaluation_total} images`:''}</span>)}{!active.length&&<span>Metrics appear automatically when an optimizer starts.</span>}</div></section>
}
