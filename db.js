/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Learning library: a small browser database of sign samples that the app
   collects automatically (no training UI). Pure logic — no DOM access,
   safe to unit-test.

   How it learns:
     - Every time the neural engine confidently reads a held letter, the
       canonical landmark vector (see knn.js) for that hold is stored.
     - Samples are grouped per letter (the "sorting") and capped at
       MAX_PER_LETTER, keeping the freshest ones — so the library reflects
       how *this* user signs today.
     - Near-duplicate holds are rejected (minDupDist) so one pose can't
       flood the library.
     - library.toModel() turns the collection into a KNN.Model, so the
       database *is* the classifier: recognition gets better as samples
       accumulate, with no retraining step.

   Persistence: IndexedDB (a real browser database) via saveToIDB /
   loadFromIDB. Everything stays in the user's own browser — samples are
   never uploaded anywhere. */
"use strict";

const LEARN = (function () {
  const DIM = 63; // canonical landmark vector length (21 points x 3)
  const MAX_PER_LETTER = 48; // cap samples per letter (freshest kept)
  const MIN_DUP_DIST = 0.35; // reject a sample this close to one we have
  const DB_NAME = "signtype-learn";
  const DB_VERSION = 1;
  const STORE = "samples";

  /* In-memory sample library. Samples: { letter, v (Float64Array 63), t }. */
  class Library {
    constructor(opts = {}) {
      this.maxPerLetter = opts.maxPerLetter || MAX_PER_LETTER;
      this.minDupDist = opts.minDupDist != null ? opts.minDupDist : MIN_DUP_DIST;
      this.samples = [];
    }

    get total() { return this.samples.length; }

    counts() {
      const c = {};
      for (const s of this.samples) c[s.letter] = (c[s.letter] || 0) + 1;
      return c;
    }

    /* Add a sample. Returns { added, reason } — reason is "ok", "dup",
       "bad-letter" or "bad-vec". Rejects near-duplicates of the same
       letter and evicts the oldest sample of that letter past the cap. */
    add(letter, v, t) {
      if (!/^[A-Z]$/.test(letter || "")) return { added: false, reason: "bad-letter" };
      if (!v || v.length !== DIM) return { added: false, reason: "bad-vec" };
      for (const s of this.samples) {
        if (s.letter === letter && KNN.dist(s.v, v) < this.minDupDist) {
          return { added: false, reason: "dup" };
        }
      }
      let n = 0, oldest = null, oldestT = Infinity;
      for (const s of this.samples) {
        if (s.letter === letter) {
          n++;
          if (s.t < oldestT) { oldestT = s.t; oldest = s; }
        }
      }
      if (n >= this.maxPerLetter && oldest) {
        this.samples.splice(this.samples.indexOf(oldest), 1);
      }
      this.samples.push({ letter, v: Float64Array.from(v), t: t != null ? t : Date.now() });
      return { added: true, reason: "ok" };
    }

    clear() { this.samples.length = 0; }

    /* Build the k-NN model that *is* the learned engine. Letters with at
       least minSamples samples compete; confidence gates live in the
       model's classify(). */
    toModel(opts = {}) {
      const m = new KNN.Model({
        k: opts.k || 7,
        minSamples: opts.minSamples != null ? opts.minSamples : 2,
      });
      for (const s of this.samples) m.add(s.letter, s.v);
      return m;
    }

    toJSON() {
      return {
        maxPerLetter: this.maxPerLetter,
        minDupDist: this.minDupDist,
        samples: this.samples.map((s) => [s.letter, Array.from(s.v), s.t]),
      };
    }

    static fromJSON(json) {
      const lib = new Library({
        maxPerLetter: json && json.maxPerLetter,
        minDupDist: json && json.minDupDist,
      });
      for (const e of (json && json.samples) || []) {
        if (!e || !/^[A-Z]$/.test(e[0] || "")) continue;
        if (!e[1] || e[1].length !== DIM) continue;
        lib.add(e[0], e[1], e[2]);
      }
      return lib;
    }
  }

  // ---------------- IndexedDB glue (browser only) ----------------

  function openIDB() {
    return new Promise((resolve, reject) => {
      if (typeof indexedDB === "undefined") {
        reject(new Error("IndexedDB unavailable"));
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error || new Error("IndexedDB open failed"));
    });
  }

  /* Replace the persisted library with the current one (idempotent). */
  async function saveToIDB(lib) {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const store = tx.objectStore(STORE);
      store.clear();
      for (const s of lib.samples) {
        store.add({ letter: s.letter, v: Array.from(s.v), t: s.t });
      }
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  /* Merge the persisted library into an (empty) in-memory one. */
  async function loadFromIDB(lib) {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).getAll();
      req.onsuccess = () => {
        for (const row of req.result || []) {
          if (!row || !row.letter || !row.v) continue;
          lib.add(row.letter, row.v, row.t);
        }
        db.close();
        resolve();
      };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  }

  /* Serialize a library to a portable JSON string (for download / shipping
     as the app's seed database). Includes a format marker so imports can
     be validated. */
  function exportJSON(lib) {
    return JSON.stringify({
      format: "signtype-learn-db",
      version: 1,
      exportedAt: new Date().toISOString(),
      ...lib.toJSON(),
    });
  }

  /* Parse a library from an exported JSON string. Throws on anything that
     isn't a SignType learning DB. */
  function importJSON(str) {
    const json = JSON.parse(str);
    if (!json || json.format !== "signtype-learn-db") {
      throw new Error("Not a SignType learning DB file");
    }
    return Library.fromJSON(json);
  }

  /* Merge every sample of source into target (dedupe + caps apply, so
     overlaps are dropped and each library's own cap is respected).
     Returns how many samples were newly added. */
  function mergeInto(target, source) {
    let added = 0;
    for (const s of source.samples) {
      if (target.add(s.letter, s.v, s.t).added) added++;
    }
    return added;
  }

  return {
    Library, saveToIDB, loadFromIDB, exportJSON, importJSON, mergeInto,
    DIM, MAX_PER_LETTER, MIN_DUP_DIST,
  };
})();