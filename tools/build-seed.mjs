/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Builds the shipped seed DB from real MediaPipe hand landmarks
   (Siruyy/asl-static-landmarks-v1, CC-BY-4.0, 4064 samples, all 26
   letters). The first 63 feature columns are raw landmarks (x/y absolute
   normalized image coords, z wrist-relative).

   The seed is validated WITHOUT the neural net (which was found to be
   trained on an unknown preprocessing and never worked on real inputs):
     1. KNN.canonicalize turns each row into a seed sample (position,
        scale, tilt and left/right hand are normalized away).
     2. Near-duplicate webcam frames collapse under the library's dedupe.
     3. Leave-one-out k-NN self-consistency: samples whose neighbors
        disagree with their label are dropped — this cleans label noise
        using the data itself and doubles as a per-letter accuracy report.

   Usage: node tools/build-seed.mjs [--include-user] */
"use strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dir = path.join(root, "tools", "data", "siruyy");
const includeUser = process.argv.includes("--include-user");

const KNN = new Function(fs.readFileSync(path.join(root, "knn.js"), "utf8") + "\nreturn KNN;")();
const LEARN = new Function("KNN", fs.readFileSync(path.join(root, "db.js"), "utf8") + "\nreturn LEARN;")(KNN);

function readNpy(p) {
  const b = fs.readFileSync(p);
  const hlen = b[6] === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const desc = b.toString("ascii", 10, 10 + hlen)
    .replace(/'/g, '"').replace(/False/g, "false").replace(/True/g, "true")
    .replace(/\(/g, "[").replace(/\)/g, "]")
    .replace(/,\s*}/g, "}").replace(/,\s*\]/g, "]");
  const { descr, shape } = JSON.parse(desc);
  const n = shape.reduce((a, c) => a * c, 1);
  const off = 10 + hlen;
  const dtMap = { "<f8": [Float64Array, 8], "<f4": [Float32Array, 4], "<i8": [BigInt64Array, 8], "<i4": [Int32Array, 4] };
  const [Ctor, size] = dtMap[descr] || [Float64Array, 8];
  const arr = new Ctor(b.buffer.slice(b.byteOffset + off, b.byteOffset + off + n * size));
  return { shape, arr };
}

const X = readNpy(path.join(dir, "X_train.npy"));
const mean = readNpy(path.join(dir, "mean.npy")).arr;
const std = readNpy(path.join(dir, "std.npy")).arr;
const Y = readNpy(path.join(dir, "y_train.npy")).arr;
const [rows, cols] = X.shape;
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const PER_LETTER = 20;   // cap samples per letter in the shipped seed

// 1. canonicalize every row (A–Y; J/Z are motion letters, not static)
const lib = new LEARN.Library();
const counts = {};
for (let i = 0; i < rows; i++) {
  const letter = ALPHABET[Number(Y[i])];
  if (!letter || letter === "J" || letter === "Z") continue;
  const lm = [];
  for (let j = 0; j < 21; j++) {
    lm.push([
      X.arr[i * cols + j * 3] * std[j * 3] + mean[j * 3],
      X.arr[i * cols + j * 3 + 1] * std[j * 3 + 1] + mean[j * 3 + 1],
      X.arr[i * cols + j * 3 + 2] * std[j * 3 + 2] + mean[j * 3 + 2],
    ]);
  }
  const v = KNN.canonicalize(lm);
  if (!v) continue;
  lib.add(letter, v, i);
}
if (includeUser) {
  // owner's own trained A samples (from their earlier export)
  try {
    const user = LEARN.importJSON(fs.readFileSync(path.join(root, "seed-user.json"), "utf8"));
    const before = lib.total;
    for (const s of user.samples) lib.add(s.letter, s.v, 1e15 + s.t);
    console.log(`Merged owner samples: ${lib.total - before} added (${user.samples.length} provided)`);
  } catch (e) { console.log("No seed-user.json — skipping owner samples:", e.message); }
}
const afterDedupe = { ...lib.counts() };
console.log(`After canonicalization + dedupe: ${lib.total} samples`);
console.log("  per letter:", Object.keys(afterDedupe).map((l) => `${l}:${afterDedupe[l]}`).join(" "));

// 2. leave-one-out self-consistency: drop samples whose neighbors disagree
const model = lib.toModel();
const drop = new Set();
let looOK = 0, looN = 0;
for (let i = 0; i < lib.samples.length; i++) {
  const s = lib.samples[i];
  // nearest neighbors excluding self (use a copy without s)
  const m = lib.toModel();
  m.samples = m.samples.filter((_, j) => j !== i);
  const r = m.classify(s.v);
  looN++;
  if (r.letter === s.letter) looOK++;
  else if (r.letter && r.conf >= 0.55 && r.nn <= 1.2) drop.add(i); // confident disagreement = bad label
}
const dropByLetter = {};
for (const i of drop) {
  const l = lib.samples[i].letter;
  dropByLetter[l] = (dropByLetter[l] || 0) + 1;
}
console.log(`\nLeave-one-out k-NN: ${looOK}/${looN} consistent (${(100 * looOK / looN).toFixed(1)}%)`);
console.log("  dropped as label noise:", Object.entries(dropByLetter)
  .map(([l, n]) => `${l}:${n}`).join(" ") || "none");

const cleaned = lib.samples.filter((_, i) => !drop.has(i));

// 3. per-letter accuracy report (leave-one-out on cleaned set)
const cl = new LEARN.Library();
for (const s of cleaned) cl.add(s.letter, s.v, s.t);
let ok = 0, n = 0;
const perLetterAcc = {};
for (let i = 0; i < cl.samples.length; i++) {
  const s = cl.samples[i];
  const m = cl.toModel();
  m.samples = m.samples.filter((_, j) => j !== i);
  const r = m.classify(s.v);
  n++;
  perLetterAcc[s.letter] = perLetterAcc[s.letter] || { n: 0, ok: 0 };
  perLetterAcc[s.letter].n++;
  if (r.letter === s.letter) { ok++; perLetterAcc[s.letter].ok++; }
}
console.log(`\nCleaned leave-one-out accuracy: ${ok}/${n} (${(100 * ok / n).toFixed(1)}%)`);
console.log("  per letter:", Object.keys(perLetterAcc).sort().map((l) =>
  `${l}:${(100 * perLetterAcc[l].ok / perLetterAcc[l].n).toFixed(0)}%`).join(" "));

// 4. write the seed (capped per letter, freshest kept)
const seed = new LEARN.Library();
const seen = {};
for (const s of cleaned) {
  if ((seen[s.letter] || 0) >= PER_LETTER) continue;
  seen[s.letter] = (seen[s.letter] || 0) + 1;
  seed.add(s.letter, s.v, s.t);
}
fs.writeFileSync(path.join(root, "seed-db.json"), LEARN.exportJSON(seed) + "\n");
console.log(`\nWrote seed-db.json: ${seed.total} samples, ${Object.keys(seed.counts()).length} letters (max ${PER_LETTER}/letter)`);