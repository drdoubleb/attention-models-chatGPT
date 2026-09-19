import {Transformer,RNG,probabilities,sample} from './engine.js';
let model=null,job=0;
const send=(type,data={},id=null)=>postMessage({type,...data,id});
const status=()=>({config:model.config,count:model.count,vocab:model.vocab,step:model.step,cut:model.cut,length:model.ids.length,history:model.history,text:model.text});
const wait=()=>new Promise(r=>setTimeout(r,0));
self.onmessage=async({data:m})=>{try{
  if(m.type==='stop'){job++;send('stopped',model?status():{},m.id);return;}
  if(m.type==='init'){job++;model=new Transformer(m.text,m.config);model.evaluate();send('ready',status(),m.id);return;}
  if(m.type==='load'){job++;const next=Transformer.restore(m.snapshot);model=next;send('ready',status(),m.id);return;}
  if(!model)throw Error('Create the model first.');
  if(m.type==='export'){send('export',{snapshot:model.snapshot()},m.id);return;}
  if(m.type==='inspect'){send('inspection',{result:model.inspect(m.prompt)},m.id);return;}
  if(m.type==='train'){
    if(!Number.isInteger(m.steps)||m.steps<1||m.steps>5000)throw Error('Choose 1–5000 steps.');
    const active=++job,start=performance.now();let report;
    for(let i=0;i<m.steps;i++){if(active!==job)return;report=model.trainStep(m.lr,4);if((i+1)%10===0||i===m.steps-1){const point=model.evaluate();send('progress',{...report,...point,elapsed:(performance.now()-start)/1000,completed:i+1,requested:m.steps},m.id);await wait();}}
    if(active===job)send('trained',status(),m.id);return;
  }
  if(m.type==='generate'){
    if(!Number.isInteger(m.length)||m.length<1||m.length>500)throw Error('Choose 1–500 generated characters.');
    const active=++job,rng=new RNG(m.seed??123);let output=m.prompt||model.vocab[model.ids[0]];model.encode(output);
    for(let i=0;i<m.length;i++){if(active!==job)return;const inspected=model.inspect(output),p=probabilities(inspected.logits,m.temperature,m.topK),index=sample(p,rng);output+=model.vocab[index];send('token',{output,char:model.vocab[index],probability:p[index],distribution:p,context:inspected.chars,completed:i+1},m.id);await new Promise(r=>setTimeout(r,m.length===1?0:16));}
    if(active===job)send('generated',{output},m.id);return;
  }
}catch(e){if(['train','generate','init','load'].includes(m.type))job++;send('error',{message:e.message},m.id);}};
