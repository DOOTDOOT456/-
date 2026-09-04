// Export the tfjs landmark MLP into ../asl-net.json for net.js.
// Reads the HF model files (tools/model/*) and slices the float32 weight
// shard per the manifest. tools/parity-tfjs.mjs proves this slicing is
// byte-exact against the real TensorFlow.js model.
import { readFileSync, writeFileSync } from "node:fs";

const spec = JSON.parse(readFileSync("model/model.json", "utf8"));
const shard = readFileSync("model/weights.bin");
const letters = JSON.parse(readFileSync("model/labels.json", "utf8"));
const f32 = new Float32Array(shard.buffer, shard.byteOffset, shard.byteLength / 4);

const manifest = spec.weightsManifest[0].weights;
let off = 0;
const weights = {};
for (const w of manifest) {
  const n = w.shape.reduce((a, b) => a * b, 1);
  const vals = Array.from(f32.subarray(off, off + n));
  off += n;
  const bare = w.name.replace(/^sequential_\d+\//, "");
  weights[bare] = { shape: w.shape, vals };
}
if (off * 4 !== shard.byteLength) {
  console.error("shard mismatch:", off * 4, "vs", shard.byteLength);
  process.exit(1);
}
console.log("consumed", off * 4, "of", shard.byteLength, "bytes");
console.log("layers:", Object.keys(weights).join(", "));

const round = (v) => (typeof v === "number" && !Number.isInteger(v) ? Number(v.toPrecision(6)) : v);
const layers = [
  { type: "dense", units: 128, activation: "relu", w: weights["dense_3/kernel"].vals, b: weights["dense_3/bias"].vals },
  { type: "dense", units: 64, activation: "relu", w: weights["dense_4/kernel"].vals, b: weights["dense_4/bias"].vals },
  { type: "dense", units: 24, activation: "softmax", w: weights["dense_5/kernel"].vals, b: weights["dense_5/bias"].vals },
];
const json = { source: "AmimulBmeIU/asl-alphabet-classifier (MIT)", letters, layers };
const body = JSON.stringify(json, (k, v) => round(v));
writeFileSync("../asl-net.json", body);
console.log("exported ../asl-net.json", (body.length / 1024).toFixed(0), "KB,", letters.length, "letters");
