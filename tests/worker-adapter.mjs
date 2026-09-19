// Node-only harness for the exact browser worker. No replacement model code.
import {parentPort} from 'node:worker_threads';
globalThis.self={};
globalThis.postMessage=message=>parentPort.postMessage(message);
await import('../dist/worker.js');
parentPort.on('message',data=>self.onmessage({data}));
