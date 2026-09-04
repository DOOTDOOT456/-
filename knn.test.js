/* Copyright © 2026 [Your Full Name]. All rights reserved.

   k-NN train-your-own classifier checks — run with: node knn.test.js
   Verifies canonicalization invariance, weighted k-NN accuracy over
   jittered synthetic poses, persistence round-trips, and the unknown-pose
   distance gate. */
"use strict";

const fs = require("fs");
const KNN = new Function(fs.readFileSync("knn.js", "utf8") + "\nreturn KNN;")();
const { POSE_CASES, jitterHand } = require("./test-hand.js");

let pass = 0, fail = 0;
function check(ok, msg) {
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${msg}`);
}

// ---------------- canonicalization invariants ----------------
{
  const base = POSE_CASES[0].hand(); // B
  const c0 = KNN.canonicalize(base);
  const maxDiff = (a, b) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  };

  // translate + scale (x/y/z) should cancel out
  const moved = base.map((p) => [0.2 + p[0] * 1.7, 0.1 + p[1] * 1.7, 0.05 + p[2] * 1.7]);
  check(maxDiff(c0, KNN.canonicalize(moved)) < 1e-6, "canonicalize: translation/scale invariant");

  // horizontal mirror (the other hand) should produce the same vector
  const mirrored = base.map((p) => [1 - p[0], p[1], p[2]]);
  check(maxDiff(c0, KNN.canonicalize(mirrored)) < 1e-6, "canonicalize: mirror (left/right hand) invariant");

  // any in-plane rotation of the whole hand must cancel out — a tilted hand
  // must canonicalize identically to an upright one (rotations about the
  // optical axis are the one freedom a flat palm has and the model must
  // normalize them away)
  const rotate = (lm, deg) => {
    const r = deg * Math.PI / 180;
    const cos = Math.cos(r), sin = Math.sin(r);
    return lm.map((p) => [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos, p[2]]);
  };
  check(maxDiff(c0, KNN.canonicalize(rotate(base, 35))) < 1e-6, "canonicalize: 35° rotation invariant");
  check(maxDiff(c0, KNN.canonicalize(rotate(base, -25))) < 1e-6, "canonicalize: -25° rotation invariant");
  check(maxDiff(c0, KNN.canonicalize(rotate(base, 180))) < 1e-6, "canonicalize: 180° rotation invariant");

  check(KNN.canonicalize(null) === null && KNN.canonicalize([]) === null,
    "canonicalize: garbage input returns null");
}

// ---------------- model basics ----------------
{
  const m = new KNN.Model({ k: 5, minSamples: 2 });
  check(m.total === 0 && m.classify([0]).letter === null && m.classify([0]).nn === Infinity,
    "model: empty model never classifies");
  const v = new Array(63).fill(0.1);
  m.add("A", v);
  m.add("A", v.map((x, i) => (i % 3 === 0 ? x + 0.01 : x)));
  m.add("B", v.map((x) => x + 0.5));
  check(m.counts().A === 2 && m.counts().B === 1 && m.total === 3, "model: counts/total");
  check(m.activeLetters().join("") === "A", "model: only A passes minSamples=2");
  const r = m.classify(v);
  check(r.letter === "A" && r.conf > 0.99 && r.active === 1, "model: classify picks majority letter");
  const last = m.removeLast();
  check(last && last.letter === "B" && m.total === 2, "model: removeLast removes newest sample");
  m.clear();
  check(m.total === 0, "model: clear");
}

// ---------------- k-NN accuracy over jittered synthetic poses ----------------
{
  const m = new KNN.Model({ k: 7, minSamples: 2 }); // defaults the app uses
  const sigma = 0.004; // ~1% of frame width of landmark noise

  for (const { name } of POSE_CASES) {
    const pose = POSE_CASES.find(c => c.name === name).hand();
    for (let i = 0; i < 4; i++) {
      m.add(name, KNN.canonicalize(jitterHand(pose, sigma, name.charCodeAt(0) * 1000 + i * 37)));
    }
  }

  let worst = { conf: 1, nn: 0, name: "" };
  let letterFails = 0;
  for (const { name } of POSE_CASES) {
    const pose = POSE_CASES.find(c => c.name === name).hand();
    for (let q = 0; q < 3; q++) {
      const lm = jitterHand(pose, sigma, name.charCodeAt(0) * 2000 + q * 101);
      const res = m.classify(KNN.canonicalize(lm));
      const ok = res.letter === name;
      if (!ok) letterFails++;
      check(ok, `knn: ${name} query #${q} -> ${res.letter} (conf ${res.conf.toFixed(2)})`);
      if (res.conf < worst.conf) worst.conf = res.conf;
      if (res.nn > worst.nn) worst.nn = res.nn;
      if (res.nn === Infinity) worst.nn = 0;
    }
  }
  console.log(`  (worst same-letter conf ${worst.conf.toFixed(3)}, worst same-letter nn ${worst.nn.toFixed(3)}, letter fails ${letterFails})`);
  check(letterFails === 0, "knn: every letter recognized on jittered queries");
}

// ---------------- unknown-pose distance gate ----------------
{
  const m = new KNN.Model(); // defaults: k 7, minSamples 2
  const a = POSE_CASES.find(c => c.name === "A");
  for (let i = 0; i < 5; i++) {
    m.add("A", KNN.canonicalize(jitterHand(a.hand(), 0.004, 900 + i)));
  }
  // A trained alone → everything wins "A" at conf 1; the app relies on the
  // nearest-neighbor distance to reject poses that are far from any sample.
  const dPose = POSE_CASES.find(c => c.name === "D").hand(); // index extended
  const resD = m.classify(KNN.canonicalize(dPose));
  const resA = m.classify(KNN.canonicalize(jitterHand(a.hand(), 0.004, 555)));
  check(resD.nn > resA.nn * 3, `knn: extended-D farther from A samples than jittered A (D nn ${resD.nn.toFixed(2)} vs A nn ${resA.nn.toFixed(2)})`);
  check(resD.nn > 0.5, "knn: unseen pose clears the app's unknown-distance gate (>0.5)");
}

// ---------------- persistence round-trip ----------------
{
  const m = new KNN.Model();
  const pose = POSE_CASES.find(c => c.name === "L").hand();
  for (let i = 0; i < 3; i++) m.add("L", KNN.canonicalize(jitterHand(pose, 0.003, i)));
  const json = JSON.stringify(m.toJSON());
  const m2 = KNN.Model.fromJSON(JSON.parse(json));
  check(m2.total === 3 && m2.activeLetters().join("") === "L", "round-trip: samples survive");
  const q = KNN.canonicalize(jitterHand(pose, 0.003, 77));
  check(m2.classify(q).letter === "L" && m2.classify(q).conf === m.classify(q).conf,
    "round-trip: identical classification after reload");

  // corrupt / hostile input must not crash or poison the model
  const junk = KNN.Model.fromJSON({ samples: [["A", [1, 2]], ["!", new Array(63).fill(0)], ["B", "nope"]] });
  check(junk.total === 0, "persistence: invalid samples are dropped");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
