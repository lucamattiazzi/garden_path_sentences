import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire('/Users/lucamattiazzi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json');
const {chromium}=require('playwright');
const root=process.cwd();
const base=process.argv[2] || 'file://'+path.join(root,'site/mech-interp/garden-path-sentences/');
const live=base.startsWith('https://');
const output=path.join(root,'reports/2026-10-12-gps/.build');
await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const checks=[],errors=[];
try {
 const page=await browser.newPage();
 await page.route('https://check.grokked.it/**',route=>route.abort());
 page.on('pageerror',e=>errors.push(e.message));
 for(const file of ['index.html','esperimenti/v3-1b/','esperimenti/v3-12b/']){
  for(const [width,theme] of [[1440,'light'],[390,'light'],[390,'dark']]){
   await page.setViewportSize({width,height:1000});
   await page.emulateMedia({colorScheme:theme,reducedMotion:'reduce'});
   const url=base+file+(file.endsWith('/')&&!live?'index.html':'');
   const response=await page.goto(url);
   if(live&&response.status()!==200)throw Error('HTTP '+response.status());
   await page.locator('img').last().waitFor({state:'attached'});
   const state=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth+1,images:[...document.images].every(i=>i.complete&&i.naturalWidth>0),imageCount:document.images.length,levels:[...document.querySelectorAll('h2')].filter(h=>/^Livello [123]/.test(h.textContent)).length,pending:/PENDING|SEGNAPOSTO|@@/.test(document.body.innerText)}));
   if(state.overflow||!state.images||state.pending||(file==='index.html'&&(state.imageCount!==8||state.levels!==3)))throw Error(JSON.stringify({file,width,theme,...state}));
   checks.push({file,width,theme,...state});
   if(file==='index.html')await page.screenshot({path:path.join(output,`report-${live?'live':'local'}-${width}-${theme}.png`)});
  }
 }
 await page.setViewportSize({width:1440,height:1000});
 await page.emulateMedia({colorScheme:'light'});
 await page.goto(base+'index.html');
 for(let i=0;i<3;i++){
  const nav=page.locator('nav a').nth(i);
  const href=await nav.getAttribute('href');
  await nav.click();
  const top=await page.locator(href).evaluate(el=>el.getBoundingClientRect().top);
  if(top<0||top>250)throw Error('Anchor not visible: '+top);
  await page.screenshot({path:path.join(output,`level-${live?'live':'local'}-${i+1}.png`)});
 }
 if(live){
  await page.goto('https://research.grokked.it/');
  await page.locator('a[href="./mech-interp/"]').click();
  await page.locator('a[href="./garden-path-sentences/"]').click();
  if(await page.locator('h1').count()!==1)throw Error('Index navigation failed');
 }
 if(errors.length)throw Error(errors.join('\n'));
 const result={date:new Date().toISOString(),base,engine:'Headless Google Chrome via Playwright',checks,errors};
 await fs.writeFile(path.join(output,`browser-${live?'live':'local'}.json`),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
}finally{await browser.close();}
