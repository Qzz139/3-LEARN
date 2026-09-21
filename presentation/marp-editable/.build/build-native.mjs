import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { finalizePresentation, resolvePresentationFont } from '/Users/zzq/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='/Users/zzq/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspaceDir='/Users/zzq/Documents/3-LEARN';
const root=path.join(workspaceDir,'presentation/marp-editable');
const tmp=path.join(root,'.build');
const skill='/Users/zzq/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const font=resolvePresentationFont({fontFamily:'PingFang SC'});
const model=JSON.parse(await fs.readFile(path.join(tmp,'layout.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
function line(slide,x,y,w,h,color,width=1){return slide.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width}});}
function nativeText(slide,e){
  const pos={...e.position};
  const isCode=e.code;
  if(e.style.fontSize===180 && /^(03|04)$/.test(e.text)){
    pos.left=930;pos.width=300;pos.height=220;
  }
  if(isCode){pos.left+=e.paddingLeft;pos.top+=e.paddingTop;pos.width-=e.paddingLeft*2;pos.height-=e.paddingTop*2;}
  const text=slide.shapes.add({geometry:'textbox',name:isCode?'Editable code block':e.text?.slice(0,45),position:pos,fill:'none',line:{fill:'none',width:0}});
  if(isCode)text.text=e.text.trimEnd();
  else {
    const paragraphs=[{runs:[]}];
    for(const r of e.runs){
      const parts=r.run.split('\n');
      for(let i=0;i<parts.length;i++){
        if(i)paragraphs.push({runs:[]});
        if(parts[i])paragraphs.at(-1).runs.push({run:parts[i],textStyle:{...r.textStyle,typeface:font}});
      }
    }
    if(e.tag==='LI')paragraphs[0].runs.unshift({run:'• '});
    if(e.text==='jeguzzi / robomaster_ros')paragraphs[0].runs=paragraphs[0].runs.map(r=>({...r,link:{uri:'https://github.com/jeguzzi/robomaster_ros',isExternal:true}}));
    if(e.text==='docs.ultralytics.com/models/yolo26/')paragraphs[0].runs=paragraphs[0].runs.map(r=>({...r,link:{uri:'https://docs.ultralytics.com/models/yolo26/',isExternal:true}}));
    text.text=paragraphs;
  }
  text.text.style={typeface:isCode?'Menlo':font,fontSize:e.style.fontSize,bold:e.style.bold,color:e.style.color,alignment:e.style.alignment,verticalAlignment:'top',autoFit:'none',wrap:isCode||e.style.fontSize===180?'none':'square',insets:{top:0,left:0,right:0,bottom:0}};
  return text;
}
for(const data of model){
  const slide=deck.slides.add();slide.background.fill=['1','8','9'].includes(data.number)?'#121416':'#fafaf8';
  // Native rectangles and lines preserve the editable academic diagram.
  for(const surface of data.surfaces){
    if(surface.fill!=='none')slide.shapes.add({geometry:'rect',position:surface.position,fill:surface.fill,line:{fill:'none',width:0}});
    const p=surface.position;
    for(const b of surface.borders){if(b.width>0){if(b.side==='top')line(slide,p.left,p.top,p.width,0,b.color,b.width);if(b.side==='bottom')line(slide,p.left,p.top+p.height,p.width,0,b.color,b.width);if(b.side==='left')line(slide,p.left,p.top,0,p.height,b.color,b.width);if(b.side==='right')line(slide,p.left+p.width,p.top,0,p.height,b.color,b.width);}}
  }
  if(data.image){slide.images.add({dataUrl:data.image.url,fit:'contain',alt:data.number==='1'?'RoboMaster EP 实物照片':'六物品分拣场地实物布局',position:data.image.position});}
  for(const e of data.elements)nativeText(slide,e);
  if(data.table){const d=data.table;const table=slide.tables.add({rows:d.values.length,columns:2,left:d.position.left,top:d.position.top,width:d.position.width,height:d.position.height,columnWidths:d.columnWidths,values:d.values});table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({color:'#d7dcda',fill:'#d7dcda',width:1});for(let r=0;r<d.values.length;r++){table.rows[r].height=d.rowHeights[r];for(let c=0;c<2;c++){const cell=table.getCell(r,c);cell.fill='#fafaf8';cell.text.style={typeface:font,fontSize:30,bold:r===0,color:'#161719'};}}}
  const n={position:{left:1200,top:670,width:40,height:28},text:data.number,runs:[{run:data.number}],style:{fontSize:18,bold:false,color:'#85888c',alignment:'right'},tag:'DIV'};nativeText(slide,n);
  slide.speakerNotes.textFrame.setText('Marp 源稿：presentation/marp-editable/slides.md。文字、代码、表格和导图采用可编辑原生对象。'+(data.number==='10'?'\nReferences: https://github.com/jeguzzi/robomaster_ros\nhttps://docs.ultralytics.com/models/yolo26/':''));
}
const candidate=path.join(tmp,'native-candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
for(let i=0;i<deck.slides.items.length;i++){
 const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`native-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const final=path.join(root,'sorting-editable.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath:final,explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[2],pythonExecutable:'/Users/zzq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','2'],fontPolicy:{basis:'design',families:[font,'Menlo']},verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,'presentation/.pptx-validation.json')});
console.log(JSON.stringify(result));
