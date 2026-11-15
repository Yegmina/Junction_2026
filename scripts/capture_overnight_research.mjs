import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
const require=createRequire(import.meta.url);
let chromium;
try{({chromium}=require('playwright-core'))}catch{({chromium}=require('/home/yehormsi/codex/pilotix-t640-flat-mount/viewer/node_modules/playwright-core'))}
const out=path.resolve('docs/presentation/assets');await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1600,height:1100},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto(process.env.AEROROZUM_RESEARCH_URL||'https://136.115.179.240/presentation/research-overnight/index.html',{waitUntil:'networkidle'});
 await page.waitForFunction(()=>document.querySelector('#live').textContent==='Live measurements');
 await page.screenshot({path:path.join(out,'overnight-research-desktop.png'),fullPage:true});
 const measured=await page.evaluate(()=>({status:document.querySelector('#stamp').textContent,robotics:document.querySelector('#reason').textContent,high:document.querySelector('#high').textContent,cpu:document.querySelector('#cpu').textContent}));
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'overnight-research-mobile.png'),fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 const receipt={at:new Date().toISOString(),url:page.url(),measured,browser_errors:errors,mobile_horizontal_overflow:overflow,metrics_injected:false};
 await fs.writeFile('docs/evidence/overnight-research-browser.json',JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify(receipt,null,2));
 if(errors.length||overflow)process.exitCode=1;
}finally{await browser.close()}
