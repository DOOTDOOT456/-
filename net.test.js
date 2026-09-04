/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Neural engine checks — run with: node net.test.js
   Verifies the shipped asl-net.json: structure, weight integrity (the
   weights were byte-sliced from the model's tfjs shard and proven to
   match the original TensorFlow.js model in tools/parity-tfjs.mjs),
   and numerical behavior of the pure-JS forward pass. */
"use strict";

const fs = require("fs");
const NET = new Function(fs.readFileSync("net.js", "utf8") + "\nreturn NET;")();
const model = NET.load(JSON.parse(fs.readFileSync("asl-net.json", "utf8")));

let pass = 0, fail = 0;
function check(ok, msg) {
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${msg}`);
}

// ---------------- structure ----------------
{
  const letters = model.letters;
  const expect = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");
  check(letters.length === 24 &&
        expect.every((l, i) => letters[i] === l),
    "model: 24 static letters A-Y (no J/Z motion letters)");
  const units = model.layers.map((l) => l.units);
  check(JSON.stringify(units) === JSON.stringify([128, 64, 24]),
    "model: dense layers 128 -> 64 -> 24");
  const acts = model.layers.map((l) => l.activation);
  check(acts[0] === "relu" && acts[1] === "relu" && acts[2] === "softmax",
    "model: relu/relu/softmax activations");
}

// ---------------- weight integrity ----------------
{
  const shapes = [[63, 128], [128, 64], [64, 24]];
  let ok = true;
  for (let i = 0; i < 3; i++) {
    const [r, c] = shapes[i];
    const l = model.layers[i];
    if (l.w.length !== r * c || l.b.length !== c) ok = false;
  }
  check(ok, "weights: kernel/bias dims match (63x128, 128x64, 64x24)");
  let finite = true, maxAbs = 0;
  for (const l of model.layers) {
    for (const v of l.w) { if (!Number.isFinite(v)) finite = false; maxAbs = Math.max(maxAbs, Math.abs(v)); }
    for (const v of l.b) { if (!Number.isFinite(v)) finite = false; maxAbs = Math.max(maxAbs, Math.abs(v)); }
  }
  check(finite && maxAbs > 0 && maxAbs < 10, `weights: all finite, sane magnitude (max ${maxAbs.toFixed(2)})`);
  // provenance sanity: this exact model published by AmimulBmeIU has these
  // parameter counts (18,008 floats = the 72,032-byte tfjs shard)
  const total = model.layers.reduce((a, l) => a + l.w.length + l.b.length, 0);
  check(total === 18008, `weights: parameter count ${total} matches the source model (18008)`);
}

// ---------------- forward pass behavior ----------------
{
  const r = NET.predict(model, new Float32Array(63).fill(0.5));
  let sum = 0;
  for (const p of r.probs) sum += p;
  check(r && r.conf > 0 && Math.abs(sum - 1) < 1e-5, "predict: flat input returns valid softmax");
  check(NET.predict(model, new Float32Array(63).fill(0.5)).conf === r.conf,
    "predict: deterministic on same input");

  let finite = true, sumOk = true;
  for (let t = 0; t < 40; t++) {
    const x = new Float32Array(63);
    for (let i = 0; i < 63; i++) x[i] = Math.random();
    const p = NET.predict(model, x);
    let s = 0;
    for (const v of p.probs) { if (!Number.isFinite(v)) finite = false; s += v; }
    if (Math.abs(s - 1) > 1e-5) sumOk = false;
  }
  check(finite, "predict: no NaN/Inf over 40 random landmark inputs");
  check(sumOk, "predict: softmax sums to 1 over 40 random inputs");
  check(NET.predict(model, new Float32Array(62)) === null &&
        NET.predict(model, null) === null && NET.predict(null, new Float32Array(63)) === null,
    "predict: bad input shapes return null (no crash)");
}

// ---------------- zero hand vs noise ----------------
{
  const z = NET.predict(model, new Float32Array(63));
  const n = NET.predict(model, new Float32Array(63).fill(0.01));
  check(z && n && z.letter === n.letter,
    "predict: near-degenerate inputs agree (calibration canary)");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
