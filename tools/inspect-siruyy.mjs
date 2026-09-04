/* Copyright © 2026 [Your Full Name]. All rights reserved.
   Inspects Siruyy/asl-static-landmarks-v1 npy files: checks whether the
   first 63 feature columns are raw MediaPipe landmarks (usable as a
   training seed after canonicalization). Run: node tools/inspect-siruyy.mjs */
"use strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "data", "siruyy");

function readNpy(p) {
  const b = fs.readFileSync(p);
  const ver = b[6];
  const hlen = ver === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const desc = b.toString("ascii", 10, 10 + hlen)
    .replace(/'/g, "\"")
    .replace(/False/g, "false").replace(/True/g, "true")
    .replace(/\(/g, "[").replace(/\)/g, "]")
    .replace(/,\s*}/g, "}")
    .replace(/,\s*\]/g, "]");
  const descr = JSON.parse(desc);
  const dtMap = { "<f8": "Float64", "<f4": "Float32", "<i8": "Int64", "<i4": "Int32", "|u1": "Uint8" };
  const dt = dtMap[descr.descr] || descr.descr;
  const shape = descr.shape;
  const n = shape.reduce((a, c) => a * c, 1);
  const off = 10 + hlen;
  const C = { Float64: Float64Array, Float32: Float32Array, Int64: BigInt64Array, Int32: Int32Array, Uint8: Uint8Array };
  const arr = new C[dt](b.buffer.slice(b.byteOffset + off, b.byteOffset + off + n * C[dt].BYTES_PER_ELEMENT));
  return { shape, arr };
}

const X = readNpy(path.join(dir, "X_train.npy"));
const M = readNpy(path.join(dir, "mean.npy"));
const S = readNpy(path.join(dir, "std.npy"));
const Y = readNpy(path.join(dir, "y_train.npy"));

console.log("X", X.shape, "mean", M.shape, "std", S.shape, "y", Y.shape);
console.log("y labels 0-19:", Array.from(Y.arr.slice(0, 20)));
console.log("y unique:", [...new Set(Array.from(Y.arr))].slice(0, 30).join(","));

const [rows, cols] = X.shape;
console.log("\ncolumn stats (raw -> unstandardized = z*std+mean):");
for (const c of [0, 1, 2, 3, 4, 5, 62]) {
  let sum = 0, mn = 0, mx = 0;
  for (let r = 0; r < rows; r++) {
    const v = X.arr[r * cols + c];
    sum += v;
    if (r === 0 || v < mn) mn = v;
    if (r === 0 || v > mx) mx = v;
  }
  const mu = sum / rows;
  const unstdMean = mu * S.arr[c] + M.arr[c];
  const unstdMin = mn * S.arr[c] + M.arr[c];
  const unstdMax = mx * S.arr[c] + M.arr[c];
  console.log(`  col ${c}: std-z range [${mn.toFixed(2)}, ${mx.toFixed(2)}] -> raw [${unstdMin.toFixed(3)}, ${unstdMax.toFixed(3)}], raw mean ${unstdMean.toFixed(3)}`);
}

// is the wrist (cols 0-2) constant (relative-to-wrist encoding)?
let wristSpread = 0;
for (const c of [0, 1, 2]) {
  let mn = Infinity, mx = -Infinity;
  for (let r = 0; r < rows; r++) {
    const v = X.arr[r * cols + c];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const spread = (mx - mn) * S.arr[c];
  console.log(`  wrist col ${c}: raw spread ${spread.toFixed(3)}`);
  wristSpread += spread;
}
console.log(wristSpread < 1 ? "  -> wrist is ~constant: features are wrist-relative (great for canonicalize)" : "  -> wrist varies: features are absolute coords (still OK — canonicalize translates)");