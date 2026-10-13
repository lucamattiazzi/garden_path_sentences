import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const runtime = process.env.GARDEN_RUNTIME ?? '/Users/lucamattiazzi/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill = process.env.GARDEN_PRESENTATION_SKILL ?? '/Users/lucamattiazzi/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
process.env.RUNTIME_NODE_MODULES ??= path.join(runtime, 'node/node_modules');
const require = createRequire(path.join(runtime, 'node/node_modules/package.json'));
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const root = path.resolve(import.meta.dirname, '..');
const repo = path.resolve(root, '../..');
const build = path.join(root, '.build');
await fs.mkdir(build, { recursive: true });
await fs.mkdir(path.join(root, 'output'), { recursive: true });
const content = JSON.parse(await fs.readFile(path.join(import.meta.dirname, 'content.json'), 'utf8'));
const baseline = JSON.parse(await fs.readFile(path.join(repo,'gemma3_4b/logs/01_baseline.json'),'utf8')).results;
const peaks = [];
for (const tag of ['1b','4b','12b','27b']) {
  const data = JSON.parse(await fs.readFile(path.join(repo,`gemma3_${tag}/logs/11_patching.json`),'utf8')).results;
  const peak = data.per_layer.reduce((a,b) => b.recovery_v_amb > a.recovery_v_amb ? b : a);
  if (data.n_pairs !== 20) throw new Error('Unexpected patching sample size');
  peaks.push([tag.toUpperCase(),`${peak.layer} / ${data.per_layer.length}`,`${Math.round(peak.recovery_v_amb*100)}%`]);
}
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = '#222222', blue = '#0000EE';
function text(s, value, x, y, width, height, size=36, options={}) {
  const box = s.shapes.add({ geometry: 'textbox', position: { left:x, top:y, width, height }, fill:'none', line:{fill:'none',width:0} });
  box.text = value;
  box.text.style = { typeface:'Arial', fontSize:size, color:ink, autoFit:'none', ...options };
  return box;
}
function table(s,values,x,y,width,height,widths) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,values,columnWidths:widths});
  t.styleOptions={headerRow:false,bandedRows:false};
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:32,color:ink},margins:{left:16,right:12,top:12,bottom:12}});
  t.borders.assign({fill:'#DDDDDD',width:1,style:'solid'});
  for(let c=0;c<values[0].length;c++)t.getCell(0,c).text.style={typeface:'Arial',fontSize:28,color:ink,bold:true};
  return t;
}
for (let i=0;i<content.length;i++) {
  const c=content[i],s=p.slides.add();
  s.background.fill='#FFFFFF';
  text(s,c.title,80,i===0?170:(i===1?275:68),1120,100,i===0?76:(i===1?66:52),{bold:true,color:'#000000'});
  if(i>0)text(s,String(i+1).padStart(2,'0'),1152,652,48,30,19,{color:'#666666'});
  s.speakerNotes.textFrame.setText(`${c.time}\n\nCANOVACCIO: ${c.cue}\n\n${c.script}\n\nPASSAGGIO: ${c.transition}\n\nFONTI: ${c.source}\nRepository: https://github.com/lucamattiazzi/garden_path_sentences\nIl materiale segue l'audit e il protocollo locali aggiornati, non tutte le interpretazioni del post iniziale.\nStile: https://grokked.it/ (tema chiaro, consultato 1 ottobre 2026).`);
  switch(i) {
    case 0:
      text(s,'Dentro un modello, davanti a una frase ambigua',84,295,1100,120,40);
      text(s,'Luca Mattiazzi',84,509,1000,48,30);
      text(s,'grokked.it',84,568,1000,48,30,{color:blue});break;
    case 1:
      break;
    case 2:
      text(s,'Attention legge le posizioni precedenti.',80,234,1120,100,43,{color:blue});
      text(s,'I token futuri non riscrivono\ngli stati delle posizioni precedenti.',80,399,1120,152,42);break;
    case 3:
      text(s,'S(parola) = −ln P(parola | prefisso)',80,239,1120,102,47,{color:blue});
      text(s,'0,10     2,30 nats\n0,01     4,61 nats',80,394,1120,138,39,{typeface:'Courier New'});
      text(s,'Esempio numerico illustrativo',80,579,1120,40,25);break;
    case 4:
      text(s,'The old man the boat.\nThe old people man the boat.',80,222,1120,150,44);
      text(s,'the     parola critica\nboat   continuazione',80,421,1120,136,38,{color:blue});
      text(s,'ΔS = S(GPS) − S(normale)',80,588,1120,42,28);break;
    case 5:
      text(s,'50 coppie / 5 costruzioni',80,215,1120,74,49,{color:blue});
      text(s,'20 coppie nome-verbo per il patching',80,319,1120,64,36);
      text(s,'Qwen3 Base    4B · 14B\nGemma 3 PT    1B · 4B · 12B · 27B',80,467,1120,131,35);break;
    case 6: {
      text(s,'Gemma 3 4B · ΔS medio alla parola critica, in nats',80,177,1120,47,28);
      const chart=s.charts.add('bar',{
        position:{left:80,top:250,width:1120,height:329},
        categories:['RR','NV','NP/S','NP/Z','COORD'],
        series:[{name:'ΔS medio (nats)',values:Object.values(baseline.per_template),fill:blue,valuesFormatCode:'0.00'}],
        barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,
        dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:'Arial',fontSize:24,fill:ink}},
        xAxis:{textStyle:{typeface:'Arial',fontSize:24,fill:ink},majorGridlines:null},
        yAxis:{min:0,max:8,majorUnit:2,numberFormatCode:'0',textStyle:{typeface:'Arial',fontSize:20,fill:ink},majorGridlines:{fill:'#DDDDDD',width:1}},
        chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'
      });
      applyPresentationChartFont(chart,{fontFamily:'Arial'});
      text(s,'Medie descrittive su 50 coppie. Variabilità fra item.',80,603,1100,40,25);break;
    }
    case 7:
      text(s,'Rimuovere un contributo.\nMisurare che cosa cambia.',80,234,1120,168,48,{color:blue});
      text(s,'Nessun circuito specifico di rianalisi stabilito.',80,487,1120,110,38);break;
    case 8:
      text(s,'Donatore     The old people man the boat.\nDestinatario The old man the boat.',80,222,1120,148,32,{typeface:'Courier New'});
      text(s,'Copia dello stato di man, layer per layer',80,424,1120,92,42,{color:blue});
      text(s,'Misura: surprisal di boat',80,567,1120,57,33);break;
    case 9:
      table(s,[['Gemma 3','Layer / profondità','Gap recuperato'],...peaks],80,208,1120,346,[280,440,400]);
      text(s,'Patch su man · 20 coppie NV · layer indicizzati da zero',80,579,1120,42,26);
      text(s,'Picchi descrittivi, selezionati sugli stessi dati',80,622,1070,38,25,{color:blue});break;
    case 10:
      text(s,'Il picco localizza una sensibilità al patch.',80,242,1120,116,46,{color:blue});
      text(s,'La decisione sulla sintassi\nresta da identificare.',80,437,1120,140,44);break;
    case 11:
      text(s,'100%',80,218,1120,164,112,{color:blue});
      text(s,'Copiare lo stato finale prima di un token\npuò riprodurre la previsione per costruzione.',80,449,1120,146,39);break;
    case 12:
      text(s,'The doctor knew the nurse had arrived.',80,222,1120,92,42);
      text(s,'Chi sapeva?     Chi era arrivato?',80,354,1120,91,44,{color:blue});
      text(s,'Ruoli invertiti. Opzioni in entrambi gli ordini.',80,484,1120,65,32);
      text(s,'Pilot preparato · validazione e inferenza ancora da fare',80,586,1120,47,26);break;
    case 13:
      text(s,'Previsione. Intervento. Interpretazione.',80,234,1120,115,45,{color:blue});
      text(s,'Un circuito di rianalisi resta da identificare.',80,403,1120,98,39);
      text(s,'grokked.it\ngithub.com/lucamattiazzi/garden_path_sentences',80,560,1120,90,26,{color:blue});break;
    case 14:
      text(s,'RR       The horse raced past the barn fell.\nNV       The old man the boat.\nNP/S   The man knew the answer was wrong.\nNP/Z   While the man hunted the deer ran.\nCOORD   John kissed Mary and her sister laughed.',80,212,1120,375,32);
      text(s,'Esempi delle strutture storiche',80,611,1100,42,25,{color:blue});break;
    case 15:
      text(s,'Gemma 27B · continuazione · 10 coppie NV “canonical”',80,204,1120,73,29);
      text(s,'1,477 nats',80,313,1120,115,71,{color:blue});
      text(s,'IC bootstrap 95%: −0,156 … 3,811\nSenza il solo item #23: 0,437 nats',80,476,1120,138,36);break;
  }
}
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++) {
  const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const finalPath=path.resolve(process.env.GARDEN_FINAL_PPTX ?? path.join(root,'output/garden-path-talk.pptx'));
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath,
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','10'],
  explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[10],requiredNativeChartOwnerSlides:[7],
  materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'powerpoint',
  fontPolicy:{basis:'design',families:['Arial','Courier New']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation.json')
});
console.log(result);
