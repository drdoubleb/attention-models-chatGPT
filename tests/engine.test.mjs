import test from 'node:test';
import assert from 'node:assert/strict';
import {Transformer,Tensor,crossEntropy,probabilities,RNG,sample} from '../dist/engine.js';
import {FROST,PATTERN} from '../dist/content.js';

const config={dim:16,context:8,heads:2,seed:42};
const close=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);

test('all parameter families pass a central-difference gradient check',()=>{
  const m=new Transformer(FROST,config),ids=m.ids.slice(0,8),targets=m.ids.slice(1,9);
  const objective=()=>crossEntropy(m.forward(ids),targets);
  objective().backward();
  for(const [name,p] of Object.entries(m.params)){
    const candidates=[0,Math.floor(p.data.length/2),p.data.length-1];
    // Also inspect the largest analytic gradient (including used embedding rows).
    candidates.push(p.grad.reduce((max,g,i)=>Math.abs(g)>Math.abs(p.grad[max])?i:max,0));
    for(const i of new Set(candidates)){
      const analytic=p.grad[i],before=p.data[i],eps=1e-5;
      p.data[i]=before+eps;const hi=objective().data[0];
      p.data[i]=before-eps;const lo=objective().data[0];p.data[i]=before;
      const numeric=(hi-lo)/(2*eps);
      assert.ok(Math.abs(analytic-numeric)<2e-6,`${name}[${i}]: analytic ${analytic}, numeric ${numeric}`);
    }
  }
});

test('causal mask prevents future leakage and normalizes each head row',()=>{
  const m=new Transformer(FROST,config),ids=m.ids.slice(0,8),a=m.forward(ids,true);
  const changed=ids.slice();changed[6]=(changed[6]+1)%m.vocab.length;changed[7]=(changed[7]+3)%m.vocab.length;
  const b=m.forward(changed,true),V=m.vocab.length;
  assert.deepEqual(Array.from(a.logits.data.slice(0,6*V)),Array.from(b.logits.data.slice(0,6*V)));
  for(let h=0;h<2;h++)for(let i=0;i<8;i++){
    let sum=0;for(let j=0;j<8;j++){const v=a.att.weights[h*64+i*8+j];sum+=v;if(j>i)assert.equal(v,0);}
    close(sum,1,1e-12);
  }
});

test('real optimization reduces loss and updates every meaningful parameter family',()=>{
  const m=new Transformer(FROST,config),first=m.evaluate(),before=Object.fromEntries(Object.entries(m.params).map(([n,p])=>[n,p.data.slice()]));
  for(let i=0;i<100;i++)m.trainStep(.003,4);
  const last=m.evaluate();assert.ok(last.train<first.train*.8,JSON.stringify({first,last}));
  for(const name of ['token','position','qw','kw','vw','ow','fw','pw','uw','ln1g','ln2g','lnfg'])
    assert.ok(m.params[name].data.some((v,i)=>Math.abs(v-before[name][i])>1e-7),name+' must update');
});

test('train and validation windows stay entirely inside their own split',()=>{
  const m=new Transformer(FROST,config),rng=new RNG(3);
  for(let i=0;i<100;i++)for(const validation of [false,true]){
    const [x,y]=m.window(rng,validation),section=validation?m.ids.slice(m.cut):m.ids.slice(0,m.cut);
    assert.equal(x.length,8);assert.deepEqual(x.slice(1),y.slice(0,-1));
    const joined=x.concat(y.at(-1)).join(',');assert.ok(section.join(',').includes(joined));
  }
});

test('checkpoint restores exact predictions and next Adam update',()=>{
  const a=new Transformer(FROST,config);for(let i=0;i<5;i++)a.trainStep();
  const b=Transformer.restore(JSON.parse(JSON.stringify(a.snapshot())));
  assert.deepEqual(a.inspect('Two roads').logits,b.inspect('Two roads').logits);
  a.trainStep();b.trainStep();for(const name of Object.keys(a.params))assert.deepEqual(a.params[name].data,b.params[name].data);
});

test('inference does not change weights, optimizer, or training random state',()=>{
  const m=new Transformer(FROST,config),before=JSON.stringify(m.snapshot()),rng=new RNG(11);let text='Two';
  for(let i=0;i<12;i++){const p=probabilities(m.inspect(text).logits,.8,5);text+=m.vocab[sample(p,rng)];}
  assert.equal(JSON.stringify(m.snapshot()),before);
});

test('stable softmax, greedy mode, top-k, and cross-entropy handle extreme scores',()=>{
  close(probabilities([1000,1001,999]).reduce((a,b)=>a+b,0),1);
  assert.deepEqual(probabilities([1,4,2],0),[0,1,0]);assert.deepEqual(probabilities([1,4,2],1,1),[0,1,0]);
  assert.equal(crossEntropy(new Tensor([1000,-1000],1,2),[1]).data[0],2000);
  assert.throws(()=>probabilities([1,2],-1));
});

test('invalid corpus, configuration, prompt and checkpoint are rejected',()=>{
  assert.throws(()=>new Transformer('too short',config));assert.throws(()=>new Transformer(FROST,{dim:17}));
  const m=new Transformer(PATTERN,config);assert.throws(()=>m.encode('🦋'));
  const snapshot=m.snapshot();snapshot.weights.qw.data[0]=null;assert.throws(()=>Transformer.restore(snapshot));
  assert.throws(()=>Transformer.restore({format:'unrelated'}));
});
