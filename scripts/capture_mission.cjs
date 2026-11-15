// Read-only presentation capture; never starts a mission or issues controls.
const {chromium}=require('/home/yehormsi/codex/pilotix-t640-flat-mount/viewer/node_modules/playwright-core');
const fs=require('fs');const path=require('path');const {spawnSync}=require('child_process');
const id=process.argv[2];if(!/^[a-f0-9]{16}$/.test(id||''))throw Error('Mission ID required');
const folder='/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/artifacts/presentation';
(async()=>{
 fs.mkdirSync(folder,{recursive:true});const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const context=await browser.newContext({viewport:{width:1500,height:1200},recordVideo:{dir:folder,size:{width:1500,height:1200}}});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const read=async()=>await(await page.request.get(`http://127.0.0.1:18901/api/missions/${id}`)).json();
 const before=await read();await page.goto(`http://127.0.0.1:18900/?mission=${id}`,{waitUntil:'domcontentloaded'});
 await page.waitForSelector('.ms-native');await page.locator('.ms-native').evaluate(e=>e.scrollIntoView({block:'start'}));
 await page.waitForTimeout(35000);await page.screenshot({path:path.join(folder,`mission-${id}-desktop.png`),fullPage:true});const after=await read();
 const video=page.video();await context.close();const raw=path.join(folder,`mission-${id}-dashboard.webm`);fs.renameSync(await video.path(),raw);await browser.close();
 const output=path.join(folder,`mission-${id}-dashboard.mp4`);const result=spawnSync('/usr/bin/ffmpeg',['-y','-loglevel','error','-i',raw,'-c:v','libx264','-preset','fast','-crf','23','-movflags','+faststart','-an',output],{encoding:'utf8'});
 if(result.status!==0)throw Error('Video conversion failed: '+result.stderr);
 const receipt={mission_id:id,read_only:true,control_requests:0,model_requests:0,before:{status:before.status,route_points:before.route.length,observations:before.observations.length},after:{status:after.status,route_points:after.route.length,observations:after.observations.length},browser_errors:errors,video:output};
 fs.writeFileSync(path.join(folder,`mission-${id}-capture.json`),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
})().catch(e=>{console.error(e);process.exit(1)});
