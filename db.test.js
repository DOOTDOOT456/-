/* Copyright © 2026 [Your Full Name]. All rights reserved.
   Learning-library checks — run with: node db.test.js */
"use strict";
const fs = require("fs");
const KNN = new Function(fs.readFileSync("knn.js", "utf8") + "\nreturn KNN;")();
// db.js uses the KNN global (as in the browser), so pass it into scope.
const LEARN = new Function("KNN", fs.readFileSync("db.js", "utf8") + "\nreturn LEARN;")(KNN);

let pass = 0, fail = 0;
function ok(cond, msg) {
  if (cond) { pass++; }
  else { fail++; console.log("FAIL:", msg); }
}
function eq(a, b, msg) {
  if (a === b) { pass++; }
  else { fail++; console.log("FAIL:", msg, "— got", a, "expected", b); }
}

// ---------------- helpers ----------------
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// random canonical-ish vector (dim 63)
function randVec(seed, jitter) {
  const rnd = mulberry32(seed);
  const v = new Float64Array(63);
  for (let i = 0; i < 63; i++) v[i] = rnd() * jitter;
  return v;
}
function jitter(base, seed, amount) {
  const rnd = mulberry32(seed);
  const v = new Float64Array(63);
  for (let i = 0; i < 63; i++) v[i] = base[i] + (rnd() - 0.5) * 2 * amount;
  return v;
}
// far-apart, deterministic vectors for cap tests (pairwise dist >= 3)
function spreadVec(i) {
  const v = new Float64Array(63);
  v[0] = i * 3;
  return v;
}
const CONF_MIN = 0.55, NN_MAX = 1.2;

// ---------------- add / counts ----------------
{
  const lib = new LEARN.Library();
  eq(lib.total, 0, "empty library");
  const r = lib.add("A", randVec(1, 1), 1000);
  eq(r.added, true, "first sample accepted");
  eq(r.reason, "ok", "reason ok");
  eq(lib.total, 1, "one sample");
  eq(lib.counts().A, 1, "counts A=1");
  ok(!lib.add("z", randVec(2, 1), 1001).added, "lowercase letter rejected");
  ok(!lib.add("AB", randVec(2, 1), 1001).added, "multi-char letter rejected");
  ok(!lib.add("A", [1, 2], 1001).added, "short vector rejected");
  ok(!lib.add("A", null, 1001).added, "null vector rejected");
  eq(lib.total, 1, "bad adds ignored");
}

// ---------------- dedupe ----------------
{
  const lib = new LEARN.Library();
  const v = randVec(10, 1);
  lib.add("A", v, 1000);
  const dup = lib.add("A", Float64Array.from(v), 2000);
  eq(dup.added, false, "identical same-letter sample rejected");
  eq(dup.reason, "dup", "reason dup");
  eq(lib.total, 1, "dup not stored");

  // same vector under a different letter is allowed (per-letter dedupe)
  lib.add("S", Float64Array.from(v), 3000);
  eq(lib.total, 2, "same shape under another letter is a new sample");

  // slightly different same-letter sample passes the threshold
  const near = jitter(v, 77, 0.03); // distance ~ 0.2 < 0.35
  eq(lib.add("A", near, 4000).added, false, "near-duplicate same-letter rejected");
  const far = jitter(v, 78, 0.5); // distance ~ 4.6 > 0.35
  eq(lib.add("A", far, 5000).added, true, "distinct same-letter sample accepted");
}

// ---------------- per-letter cap keeps freshest ----------------
{
  const lib = new LEARN.Library({ maxPerLetter: 10 });
  for (let i = 0; i < 20; i++) lib.add("A", spreadVec(i), 1000 + i);
  eq(lib.total, 10, "capped at maxPerLetter");
  eq(lib.counts().A, 10, "letter count capped");
  const ts = lib.samples.map((s) => s.t).sort((a, b) => a - b);
  eq(ts[0], 1010, "oldest evicted, freshest kept (t starts at 1010)");
  eq(ts[9], 1019, "newest kept");

  // a second letter is unaffected by the cap
  lib.add("B", spreadVec(0), 5000);
  lib.add("B", spreadVec(1), 5001);
  eq(lib.counts().B, 2, "B uncapped");
  eq(lib.total, 12, "total across letters");
}

// ---------------- toModel: the library *is* the classifier ----------------
{
  const lib = new LEARN.Library();
  const cA = randVec(1, 1), cS = randVec(2, 1);
  // realistic hold-to-hold jitter (distinct samples must clear 0.35)
  for (let i = 0; i < 3; i++) lib.add("A", jitter(cA, 100 + i, 0.2), 1000 + i);
  for (let i = 0; i < 3; i++) lib.add("S", jitter(cS, 200 + i, 0.2), 2000 + i);
  const model = lib.toModel();
  eq(model.total, 6, "model holds all samples");
  eq(model.activeLetters().join(""), "AS", "both letters compete");

  const qA = model.classify(jitter(cA, 900, 0.05));
  eq(qA.letter, "A", "A query → A");
  ok(qA.conf >= CONF_MIN, "A confidence above gate");
  ok(qA.nn <= NN_MAX, "A nearest-neighbor within gate");

  const qS = model.classify(jitter(cS, 901, 0.05));
  eq(qS.letter, "S", "S query → S");
  ok(qS.conf >= CONF_MIN, "S confidence above gate");

  // an unfamiliar pose is rejected by the nn gate
  const qFar = model.classify(randVec(500, 3));
  ok(qFar.nn > NN_MAX, "unseen pose far from all samples");

  // a lone sample never competes (minSamples = 2)
  lib.add("Q", jitter(cA, 902, 0.05), 9999);
  const m2 = lib.toModel();
  eq(m2.classify(jitter(cA, 903, 0.05)).letter, "A", "Q with 1 sample ignored");
}

// ---------------- JSON round-trip ----------------
{
  const lib = new LEARN.Library();
  const cA = randVec(3, 1), cS = randVec(4, 1);
  lib.add("A", jitter(cA, 1, 0.05), 1000);
  lib.add("A", jitter(cA, 2, 0.05), 1001);
  lib.add("S", jitter(cS, 3, 0.05), 2000);
  const back = LEARN.Library.fromJSON(JSON.parse(JSON.stringify(lib.toJSON())));
  eq(back.total, 3, "round-trip keeps samples");
  eq(back.counts().A, 2, "round-trip counts A");
  eq(back.counts().S, 1, "round-trip counts S");
  const m = back.toModel();
  eq(m.classify(jitter(cA, 4, 0.05)).letter, "A", "round-trip classifies");

  // sanitizing: junk rows are skipped
  const dirty = LEARN.Library.fromJSON({
    samples: [["1", randVec(9, 1), 1], ["A", [1, 2], 1], ["A", randVec(8, 1), 1], ["?", null, 1]],
  });
  eq(dirty.total, 1, "dirty JSON sanitized");
  eq(dirty.counts().A, 1, "only valid row kept");
}

// ---------------- IndexedDB persistence (mock backend) ----------------
{
  // Minimal fake IndexedDB with a row array backing store.
  function fakeIndexedDB() {
    const rows = [];
    let nextId = 1;
    const open = () => {
      const req = {};
      queueMicrotask(() => {
        const db = {
          objectStoreNames: { contains: () => true },
          createObjectStore: () => {},
          transaction: () => {
            const tx = { oncomplete: null, onerror: null };
            tx.objectStore = () => {
              return {
                clear: () => { rows.length = 0; },
                add: (row) => { row.id = nextId++; rows.push(row); },
                getAll: () => {
                  const r = {};
                  queueMicrotask(() => { r.result = rows.map((x) => ({ letter: x.letter, v: x.v, t: x.t })); });
                  return r;
                },
              };
            };
            queueMicrotask(() => { if (tx.oncomplete) tx.oncomplete(); });
            return tx;
          },
          close: () => {},
        };
        req.result = db;
        if (req.onupgradeneeded) req.onupgradeneeded();
        if (req.onsuccess) req.onsuccess();
      });
      return req;
    };
    return { open, rows };
  }

  const fake = fakeIndexedDB();
  const realIndexedDB = global.indexedDB;
  global.indexedDB = fake;

  const lib = new LEARN.Library();
  const cA = randVec(5, 1), cS = randVec(6, 1);
  lib.add("A", jitter(cA, 10, 0.2), 1000);
  lib.add("A", jitter(cA, 11, 0.2), 1001);
  lib.add("S", jitter(cS, 12, 0.2), 2000);

  (async () => {
    await LEARN.saveToIDB(lib);
    eq(fake.rows.length, 3, "IDB store has 3 rows");
    const back = new LEARN.Library();
    await LEARN.loadFromIDB(back);
    eq(back.total, 3, "loaded library matches");
    eq(back.counts().A, 2, "loaded counts A");
    eq(back.toModel().classify(jitter(cA, 13, 0.05)).letter, "A", "loaded library classifies");

    // save is idempotent (clear + re-add)
    await LEARN.saveToIDB(lib);
    eq(fake.rows.length, 3, "second save replaces, not appends");

    global.indexedDB = realIndexedDB;
  })().then(() => {}, (e) => { fail++; console.log("FAIL: async persistence —", e.message); });
}

// ---------------- legacy My-signs migration ----------------
{
  // legacy vectors are plain arrays in localStorage, so round-trip via JSON
  const legacy = new KNN.Model({ k: 7, minSamples: 2 });
  const a0 = Array.from(randVec(20, 1));
  legacy.add("A", a0);
  legacy.add("A", Array.from(jitter(a0, 21, 0.2)));
  legacy.add("B", Array.from(randVec(22, 1)));

  // app-side migration: parse legacy JSON, pour its samples into a Library
  const restored = KNN.Model.fromJSON(JSON.parse(JSON.stringify(legacy.toJSON())));
  const lib = new LEARN.Library();
  for (const s of restored.samples) lib.add(s.letter, s.v, Date.now());
  eq(lib.total, 3, "legacy model samples migrated");
  eq(lib.counts().A, 2, "migrated counts A");
  eq(lib.counts().B, 1, "migrated counts B");
}

// ---------------- export / import / merge (the seed workflow) -------------
{
  const lib = new LEARN.Library();
  const cA = randVec(30, 1), cS = randVec(31, 1);
  lib.add("A", jitter(cA, 40, 0.2), 1000);
  lib.add("A", jitter(cA, 41, 0.2), 1001);
  lib.add("S", jitter(cS, 42, 0.2), 2000);

  const exported = LEARN.exportJSON(lib);
  const parsed = JSON.parse(exported);
  eq(parsed.format, "signtype-learn-db", "export carries format marker");
  ok(exported.length > 0, "export is a non-empty string");

  const back = LEARN.importJSON(exported);
  eq(back.total, 3, "import restores samples");
  eq(back.counts().A, 2, "import counts A");
  eq(back.toModel().classify(jitter(cA, 43, 0.05)).letter, "A", "imported DB classifies");

  // imports reject junk
  let threw = false;
  try { LEARN.importJSON(JSON.stringify({ hello: 1 })); } catch (e) { threw = true; }
  ok(threw, "non-SignType JSON rejected");
  threw = false;
  try { LEARN.importJSON("not json"); } catch (e) { threw = true; }
  ok(threw, "invalid JSON rejected");

  // merging a seed into a visitor's library adds distinct samples
  const visitor = new LEARN.Library();
  visitor.add("A", jitter(cA, 44, 0.2), 3000);
  const added = LEARN.mergeInto(visitor, back);
  eq(added, 3, "merge adds seed samples that aren't near-duplicates");
  eq(visitor.counts().A, 3, "visitor A grew (own + 2 distinct seed A)");
  eq(visitor.counts().S, 1, "visitor gained S from seed");
  eq(visitor.total, 4, "visitor total after merge");

  // merging is idempotent
  eq(LEARN.mergeInto(visitor, back), 0, "second merge adds nothing");
}

// wait for async persistence test to finish before reporting
setTimeout(() => {
  console.log(`\ndb.test.js: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}, 50);