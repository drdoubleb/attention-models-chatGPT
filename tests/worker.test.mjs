import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {FROST} from '../dist/content.js';

test('inspection errors do not cancel training; pause returns authoritative state',async t=>{
  const worker=new Worker(new URL('./worker-adapter.mjs',import.meta.url));t.after(()=>worker.terminate());
  const all=[],waiters=[];worker.on('message',m=>{all.push(m);for(const x of [...waiters])if(x.predicate(m)){waiters.splice(waiters.indexOf(x),1);clearTimeout(x.timer);x.resolve(m);}});
  const wait=predicate=>new Promise((resolve,reject)=>{const old=all.find(predicate);if(old)return resolve(old);const x={predicate,resolve,timer:setTimeout(()=>reject(Error('Timed out waiting for worker')),20000)};waiters.push(x);});
  worker.postMessage({type:'init',id:1,text:FROST,config:{dim:16,heads:2,context:8}});await wait(m=>m.type==='ready');
  worker.postMessage({type:'train',id:2,steps:30,lr:.003});worker.postMessage({type:'inspect',id:3,prompt:'🦋'});
  await wait(m=>m.type==='error'&&m.id===3);const trained=await wait(m=>m.type==='trained'&&m.id===2);assert.equal(trained.step,30);
  worker.postMessage({type:'train',id:4,steps:1000,lr:.003});await wait(m=>m.type==='progress'&&m.id===4);
  worker.postMessage({type:'stop',id:5});const stopped=await wait(m=>m.type==='stopped'&&m.id===5);
  assert.ok(stopped.step>=40&&stopped.step<1030);assert.equal(stopped.history.at(-1).step,stopped.step);
  worker.postMessage({type:'export',id:6});const exported=await wait(m=>m.type==='export'&&m.id===6);assert.equal(exported.snapshot.step,stopped.step);
  worker.postMessage({type:'generate',id:7,prompt:'Two',length:5,temperature:.8,topK:5,seed:12});const generated=await wait(m=>m.type==='generated'&&m.id===7);assert.equal(Array.from(generated.output).length,8);
  worker.postMessage({type:'export',id:8});const after=await wait(m=>m.type==='export'&&m.id===8);assert.deepEqual(after.snapshot.weights,exported.snapshot.weights);
});
