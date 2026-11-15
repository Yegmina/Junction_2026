#!/usr/bin/env node
/** Capture actual local campaign interactions; never inserts fictional result values. */
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
let chromium;
try {({chromium}=require(process.env.PLAYWRIGHT_CORE_MODULE||'playwright-core'));}
catch {({chromium}=require('/home/yehormsi/codex/pilotix-t640-flat-mount/viewer/node_modules/playwright-core'));}
const project=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(project,'docs/presentation/assets');
const base=process.env.AEROROZUM_DASHBOARD_URL||'http://127.0.0.1:18900';
const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--enable-webgl','--use-gl=angle','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1920,height:1260},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try {
 await fs.mkdir(output,{recursive:true});
 await page.goto(base,{waitUntil:'domcontentloaded'});
 await page.getByText('Research API connected',{exact:true}).waitFor();
 // The requested capture protocol uses a real 4 m launch. This changes the
 // outgoing action arguments, never the returned state, image, or screenshot.
 await page.route('**/api/campaigns/*/actions',async route=>{
  const request=route.request();const payload=request.postDataJSON();
  if(payload?.action==='launch')return route.continue({postData:JSON.stringify({...payload,arguments:{...payload.arguments,altitude_m:4}})});
  return route.continue();
 });
 await page.getByRole('button',{name:'New campaign',exact:true}).click();
 await page.getByRole('button',{name:'Launch',exact:true}).click();
 await page.getByRole('button',{name:'Capture',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.thermal-preview img')?.naturalWidth===640);
 const state=await page.evaluate(async()=>{
  const all=await(await fetch('/api/status')).json();
  const selected=document.querySelector('[aria-label="Active campaign"]')?.value;
  return all.campaigns.find(c=>c.id===selected)||all.campaigns.filter(c=>!c.read_only).at(-1);
 });
 assert.equal(state.pose.altitude_m,4);assert.ok(state.observations.length>=1);assert.equal(state.mode,'demo');
 await page.waitForTimeout(750);
 await page.screenshot({path:path.join(output,'07-dashboard-current.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(output,'07-dashboard-mobile.png'),fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 assert.equal(overflow,false,'Mobile page should not overflow horizontally');
 await page.setViewportSize({width:1920,height:1260});
 const external=await page.evaluate(async()=>{const s=await(await fetch('/api/status')).json();return s.campaigns.filter(c=>c.read_only&&c.observations?.length).at(-1)});
 let agentImage=false;
 if(external&&await page.locator('[aria-label="Active campaign"]').count()){
  await page.getByLabel('Active campaign').selectOption(external.id);
  await page.getByRole('button',{name:'Evidence',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.evidence-card img')?.naturalWidth>0);
  await page.screenshot({path:path.join(output,'08-agent-evidence.png'),fullPage:true});agentImage=true;
 }
 assert.deepEqual(errors,[],'No browser page errors expected');
 const now=new Date().toISOString();
 const entry=`\n### ${now} — Current dashboard capture\n\n- Executed a real local demonstration campaign through the dashboard: create → launch at **4 m** → acquire an original **640 × 512** thermal observation. The frame and pose were verified from the API; no image or metric was injected.\n- Saved [desktop](assets/07-dashboard-current.png) and [mobile](assets/07-dashboard-mobile.png) captures. Mobile viewport 390 × 844 had no horizontal overflow; the browser reported no page errors.\n- Fidelity: **${state.fidelity}**. This is an explicit development demonstration, not an accepted direct-physics result.\n${agentImage?'- Also captured [actual API-agent evidence](assets/08-agent-evidence.png) from a completed external campaign, displayed read-only.\n':''}`;
 await fs.appendFile(path.join(project,'docs/presentation/journal.md'),entry);
 console.log(JSON.stringify({captured_at:now,campaign_id:state.id,altitude_m:state.pose.altitude_m,observation_count:state.observations.length,fidelity:state.fidelity,agent_evidence_screenshot:agentImage,browser_errors:errors,mobile_overflow:overflow},null,2));
}finally {await browser.close();}
