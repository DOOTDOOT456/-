// Parity: hand-sliced MLP forward vs the real tfjs LayersModel (wasm).
// Isolates weight-slicing bugs from preprocessing questions.
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";
import { readFileSync } from "node:fs";
await tf.setBackend("wasm");
await tf.ready();

const spec = JSON.parse(readFileSync("model/model.json", "utf8"));
const shard = readFileSync("model/weights.bin");

// Keras 3 writes batch_shape where tfjs expects batchInputShape
const topo = JSON.parse(JSON.stringify(spec.modelTopology));
const inputLayer = topo.model_config.config.layers[0];
if (inputLayer.config.batch_shape) {
  inputLayer.config.batchInputShape = inputLayer.config.batch_shape;
  delete inputLayer.config.batch_shape;
}

const handler = {
  load: async () => ({
    modelTopology: topo,
    // tfjs strips the sequential model prefix from its internal names
    weightSpecs: spec.weightsManifest[0].weights.map((w) => ({ ...w, name: w.name.replace(/^sequential_\d+\//, "") })),
    weightData: shard.buffer.slice(shard.byteOffset, shard.byteOffset + shard.byteLength),
  }),
};
const model = await tf.loadLayersModel(handler);

// my JS forward (same as validate-model)
const f32 = new Float32Array(shard.buffer, shard.byteOffset, shard.byteLength / 4);
let off = 0;
const next = (n) => { const o = Array.from(f32.subarray(off, off + n)); off += n; return o; };
const W1 = next(63 * 128), B1 = next(128), W2 = next(128 * 64), B2 = next(64), W3 = next(64 * 24), B3 = next(24);
function dense(x, W, outN, B) {
  const out = new Array(outN).fill(0);
  for (let o = 0; o < outN; o++) { let a = B[o]; for (let i = 0; i < x.length; i++) a += x[i] * W[i * outN + o]; out[o] = a; }
  return out;
}
const relu = (x) => x.map((v) => (v > 0 ? v : 0));
function myForward(x) {
  let h = relu(dense(x, W1, 128, B1));
  h = relu(dense(h, W2, 64, B2));
  const logits = dense(h, W3, 24, B3);
  const mx = Math.max(...logits);
  const p = logits.map((v) => Math.exp(v - mx));
  const s = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / s);
}

let maxDiff = 0, bad = 0;
for (let t = 0; t < 30; t++) {
  const x = Array.from({ length: 63 }, () => Math.random() * 2 - 1);
  const mine = myForward(x);
  const theirs = model.predict(tf.tensor2d([x], [1, 63])).dataSync();
  for (let i = 0; i < 24; i++) {
    const d = Math.abs(mine[i] - theirs[i]);
    if (d > maxDiff) maxDiff = d;
    if (d > 1e-4) bad++;
  }
}
console.log("parity: max diff", maxDiff.toExponential(2), "elements over 1e-4:", bad);
