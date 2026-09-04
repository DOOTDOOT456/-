/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Audits a SignType learning DB (seed-db.json): for every sample it
   reports distance to the letter's reference pose (from test-hand.js),
   distance to the letter cluster centroid, nearest-neighbor distance, and
   per-finger curl (tip pulled toward the wrist = curled, like a fist).

   Usage: node tools/audit-db.mjs [path-to-db.json] */
"use strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dbPath = path.join(root, process.argv[2] || "seed-db.json");

const KNN = new Function(fs.readFileSync(path.join(root, "knn.js"), "utf8") + "\nreturn KNN;")();
const LEARN = new Function("KNN", fs.readFileSync(path.join(root, "db.js"), "utf8") + "\nreturn LEARN;")(KNN);
const { POSE_CASES } = require(path.join(root, "test-hand.js"));

const db = JSON.parse(fs.readFileSync(dbPath, "utf8"));
if (db.format !== "signtype-learn-db") {
  console.error("Not a SignType learning DB:", dbPath);
  process.exit(1);
}
const lib = LEARN.Library.fromJSON(db);
console.log(`Auditing ${dbPath}: ${lib.total} samples, ${Object.keys(lib.counts()).join(", ")}`);

// reference poses per letter present in the DB
const refs = {};
for (const { name, hand } of POSE_CASES) {
  refs[name] = KNN.canonicalize(hand());
}

const FINGERS = [
  ["index", 5], ["middle", 9], ["ring", 13], ["pinky", 17],
];
function curlOf(v, mcpIdx) {
  return v[(mcpIdx + 3) * 3 + 1] - v[mcpIdx * 3 + 1]; // tipY - mcpY (>0 = curled)
}

for (const letter of Object.keys(lib.counts()).sort()) {
  const samples = lib.samples.filter((s) => s.letter === letter);
  const centroid = new Float64Array(63);
  for (const s of samples) for (let i = 0; i < 63; i++) centroid[i] += s.v[i] / samples.length;
  const ref = refs[letter];

  console.log(`\n=== ${letter} (${samples.length} samples) ===`);
  if (ref) {
    console.log(`  reference ${letter} curl: ` + FINGERS.map(([n, i]) => `${n}=${curlOf(ref, i).toFixed(2)}`).join(" "));
  }
  samples.forEach((s, idx) => {
    const dRef = ref ? KNN.dist(s.v, ref).toFixed(3) : "-";
    const dCen = KNN.dist(s.v, centroid).toFixed(3);
    let dNN = Infinity;
    for (const t of samples) {
      if (t === s) continue;
      dNN = Math.min(dNN, KNN.dist(s.v, t.v));
    }
    const curl = FINGERS.map(([n, i]) => curlOf(s.v, i)).map((c) => c.toFixed(2)).join(" ");
    console.log(
      `  #${idx + 1}  refDist ${dRef}  centDist ${dCen}  nn ${dNN === Infinity ? "-" : dNN.toFixed(3)}` +
      `  curl(${curl})` +
      (dNN < LEARN.MIN_DUP_DIST ? "  <-- near-dup!" : "") +
      (ref && KNN.dist(s.v, ref) > 1.0 ? "  <-- far from reference" : "")
    );
  });
}