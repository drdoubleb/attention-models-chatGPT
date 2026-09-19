// Recreate the bundled, deliberately overfit example with the same browser engine.
import fs from 'node:fs';
import {Transformer} from '../dist/engine.js';
import {FROST} from '../dist/content.js';
const model=new Transformer(FROST);
model.evaluate();
for(let i=0;i<1200;i++){
  model.trainStep(.003,4);
  if((i+1)%10===0)model.evaluate();
}
fs.writeFileSync(new URL('../dist/poem-checkpoint.json',import.meta.url),JSON.stringify(model.snapshot()));
console.log({parameters:model.count,initial:model.history[0],final:model.history.at(-1)});
