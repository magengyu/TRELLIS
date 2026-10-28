const $ = id => document.getElementById(id);
let images = [], preview = null, busy = false;
const camera = {eye:{x:1.5,y:1.5,z:1.1},up:{x:0,y:0,z:1}};
function message(text, error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function showImages(){
  $('image-count').textContent=`${images.length} 张`;
  $('thumbnails').replaceChildren();
  images.forEach((file,index)=>{
    const item=document.createElement('div');item.className='thumb';
    const img=document.createElement('img');img.alt=file.name;
    const url=URL.createObjectURL(file);img.src=url;img.onload=()=>URL.revokeObjectURL(url);
    const remove=document.createElement('button');remove.textContent='×';remove.title=`移除 ${file.name}`;
    remove.onclick=()=>{if(!busy){images.splice(index,1);showImages();}};
    item.append(img,remove);$('thumbnails').append(item);
  });
}
function addFiles(files){
  if(busy)return;
  const next=[...images,...files];
  if(next.length>12){message('最多支持 12 张图片。',true);return;}
  if(next.some(f=>!f.type.startsWith('image/'))){message('请选择图片文件。',true);return;}
  if(next.reduce((sum,f)=>sum+f.size,0)>32*1024*1024){message('图片总大小不能超过 32 MB。',true);return;}
  images=next;showImages();message('图片已就绪，可开始生成。');
}
$('images').onchange=e=>{addFiles(e.target.files);e.target.value='';};
$('clear').onclick=()=>{if(!busy){images=[];showImages();}};
['dragenter','dragover'].forEach(event=>$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('drag');}));
['dragleave','drop'].forEach(event=>$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('drag');}));
$('dropzone').addEventListener('drop',e=>addFiles(e.dataTransfer.files));
function voxelEdges(centers){
  // Integer grid coordinates give shared edges identical keys, without float rounding.
  const seen=new Set(),x=[],y=[],z=[];
  const key=point=>point[0]+65*(point[1]+65*point[2]);
  for(const center of centers){
    const cell=center.map(value=>Math.round((value+0.5)*64-0.5));
    for(let axis=0;axis<3;axis++){
      const other=[0,1,2].filter(value=>value!==axis);
      for(let a=0;a<=1;a++)for(let b=0;b<=1;b++){
        const first=[...cell];first[other[0]]+=a;first[other[1]]+=b;
        const edgeKey=key(first)*3+axis;
        if(seen.has(edgeKey))continue;
        seen.add(edgeKey);
        const last=[...first];last[axis]++;
        x.push(first[0]/64-0.5,last[0]/64-0.5,null);
        y.push(first[1]/64-0.5,last[1]/64-0.5,null);
        z.push(first[2]/64-0.5,last[2]/64-0.5,null);
      }
    }
  }
  return {x,y,z};
}
async function draw(){
  if(!preview)return;
  const v=preview.centers;
  const traces=[];
  if($('edges').checked){
    preview.edges ??= voxelEdges(v);
    traces.push({type:'scatter3d',mode:'lines',...preview.edges,
      line:{color:'#508875',width:1},opacity:0.35,hoverinfo:'skip',showlegend:false});
  }
  if($('points').checked)traces.push({
    type:'scatter3d',mode:'markers',x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),
    marker:{size:2,color:v.map(p=>p[2]),colorscale:[[0,'#214b41'],[1,'#adc5a0']]},
    hoverinfo:'skip',showlegend:false
  });
  const axis={range:[-.52,.52],showbackground:false,gridcolor:'#e0e7de',zeroline:false,tickfont:{size:9,color:'#94a397'},title:''};
  await Plotly.react('plot',traces,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'rgba(0,0,0,0)',
    scene:{xaxis:{...axis},yaxis:{...axis},zaxis:{...axis},aspectmode:'cube',camera,bgcolor:'rgba(0,0,0,0)'},
    uirevision:'occupancy'},{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['toImage']});
  $('empty').style.display='none';
}
$('points').onchange=()=>draw();
$('edges').onchange=()=>draw();
$('reset').onclick=()=>{if(preview)Plotly.relayout('plot',{'scene.camera':camera});};
async function jsonResponse(response){const data=await response.json();if(!response.ok)throw Error(data.error||'请求失败');return data;}
$('generate').onclick=async()=>{
  if(!images.length){message('请先选择至少一张图片。',true);return;}
  for(const id of ['steps','seed','cfg'])if(!$(id).checkValidity()){message('请检查采样步数、种子和引导强度的范围。',true);return;}
  const form=new FormData();images.forEach(file=>form.append('images',file));
  for(const id of ['steps','seed','cfg','mode'])form.append(id,$(id).value);
  busy=true;$('generate').disabled=true;$('generate').textContent='正在生成…';
  $('download').classList.add('disabled');$('download').setAttribute('aria-disabled','true');
  message('正在上传图片…');
  try{
    const {id}=await jsonResponse(await fetch('/api/generate',{method:'POST',body:form}));
    while(true){
      const job=await jsonResponse(await fetch(`/api/jobs/${id}`));message(job.message);
      if(job.status==='error')throw Error(job.message);
      if(job.status==='done'){
        preview=await jsonResponse(await fetch(`/api/jobs/${id}/preview`));
        await draw();
        $('stats').textContent=`${job.count.toLocaleString()} 个占据体素 · 拖动旋转 / 滚轮缩放`;
        $('download').href=`/api/jobs/${id}/download`;$('download').classList.remove('disabled');$('download').setAttribute('aria-disabled','false');
        message(job.count?'生成完成，结果已保存。':'生成完成，但占据网格为空；请尝试其他种子或图片。');break;
      }
      await new Promise(resolve=>setTimeout(resolve,1200));
    }
  }catch(error){message(error.message,true);}
  finally{busy=false;$('generate').disabled=false;$('generate').innerHTML='生成占据网格 <span>↗</span>';}
};
