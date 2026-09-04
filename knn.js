/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Train-your-own sign classifier: k-NN over canonicalized MediaPipe hand
   landmarks. Pure logic — no DOM access, safe to unit-test.

   Each recorded sample is a MediaPipe-style flat landmark list (21 points)
   turned into one canonical vector of 63 numbers by canonicalize():
     - wrist translated to the origin,
     - scaled so wrist -> middle-MCP has length 1,
     - rotated so that axis points up on screen,
     - x-mirrored so the index-finger side is always to the right.
   That makes samples comparable no matter where the hand is in the frame,
   how big it is, or which hand signs. Classification is weighted k-NN:
   the vote weight of a neighbor is 1 / distance^2, and confidence is the
   winning letter's share of the total weight. */
"use strict";

const KNN = (function () {
  const NUM_LANDMARKS = 21;
  const DIM = NUM_LANDMARKS * 3; // 63

  /* Flat landmark list (entries {x,y,z} or [x,y,z]) -> canonical 63-vector,
     or null when the hand can't be aligned (degenerate geometry). */
  function canonicalize(raw) {
    if (!raw || raw.length < NUM_LANDMARKS) return null;
    const lm = [];
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const p = raw[i];
      lm.push(Array.isArray(p) ? [p[0], p[1], p[2]] : [p.x, p.y, p.z]);
    }
    const wrist = lm[0];
    const midMcp = lm[9];
    const dx = midMcp[0] - wrist[0];
    const dy = midMcp[1] - wrist[1];
    const s = Math.hypot(dx, dy);
    if (!(s > 1e-6)) return null;

    // Rotate so wrist -> middle-MCP points up (negative y in image coords).
    const phi = -Math.atan2(dy, dx) - Math.PI / 2;
    const cos = Math.cos(phi), sin = Math.sin(phi);

    const v = new Array(DIM);
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const px = lm[i][0] - wrist[0];
      const py = lm[i][1] - wrist[1];
      const pz = lm[i][2] - wrist[2];
      v[i * 3] = (px * cos - py * sin) / s;
      v[i * 3 + 1] = (px * sin + py * cos) / s;
      v[i * 3 + 2] = pz / s;
    }
    // Mirror to a canonical handedness: index MCP on the positive-x side.
    if (v[5 * 3] < 0) {
      for (let i = 0; i < NUM_LANDMARKS; i++) v[i * 3] = -v[i * 3];
    }
    return v;
  }

  function dist(a, b) {
    let d2 = 0;
    for (let i = 0; i < DIM; i++) {
      const t = a[i] - b[i];
      d2 += t * t;
    }
    return Math.sqrt(d2);
  }

  class Model {
    constructor(opts = {}) {
      this.k = opts.k || 7;                 // neighbors to vote
      this.minSamples = opts.minSamples || 2; // a letter must have >= this many
      this.samples = [];                    // [{letter, v}]
    }

    add(letter, v) {
      if (!letter || !v || v.length !== DIM) return;
      this.samples.push({ letter, v });
    }

    removeLast() {
      return this.samples.pop() || null;
    }

    clear() { this.samples.length = 0; }

    get total() { return this.samples.length; }

    counts() {
      const c = {};
      for (const s of this.samples) c[s.letter] = (c[s.letter] || 0) + 1;
      return c;
    }

    /* Letters that have collected at least minSamples samples. */
    activeLetters() {
      const c = this.counts();
      return Object.keys(c).filter((l) => c[l] >= this.minSamples).sort();
    }

    /* Weighted k-NN vote. Letters below minSamples never compete.
       Returns { letter, conf, nn, active }:
         letter  winning letter or null,
         conf    winning share of total vote weight in [0, 1],
         nn      distance to the single nearest neighbor (Infinity if none),
         active  number of letters that were allowed to compete. */
    classify(v) {
      const counts = this.counts();
      const active = new Set(Object.keys(counts).filter(
        (l) => counts[l] >= this.minSamples));
      const cand = [];
      for (const s of this.samples) if (active.has(s.letter)) cand.push(s);
      if (!cand.length || !v || v.length !== DIM) {
        return { letter: null, conf: 0, nn: Infinity, active: active.size };
      }

      const ranked = cand
        .map((s) => ({ s, d: dist(v, s.v) }))
        .sort((a, b) => a.d - b.d);
      const k = Math.min(this.k, ranked.length);

      const votes = {};
      let weightSum = 0;
      for (let i = 0; i < k; i++) {
        const d = ranked[i].d;
        const w = 1 / (d * d + 1e-6);
        votes[ranked[i].s.letter] = (votes[ranked[i].s.letter] || 0) + w;
        weightSum += w;
      }
      let best = null, bestW = -1;
      for (const l of Object.keys(votes)) {
        if (votes[l] > bestW) { bestW = votes[l]; best = l; }
      }
      return {
        letter: best,
        conf: weightSum > 0 ? votes[best] / weightSum : 0,
        nn: ranked[0].d,
        active: active.size,
      };
    }

    toJSON() {
      return {
        k: this.k,
        minSamples: this.minSamples,
        samples: this.samples.map((s) => [s.letter, s.v]),
      };
    }

    static fromJSON(json) {
      const m = new Model({ k: json && json.k, minSamples: json && json.minSamples });
      const list = (json && json.samples) || [];
      for (const entry of list) {
        const letter = entry && entry[0];
        const v = entry && entry[1];
        if (!/^[A-Z]$/.test(letter || "")) continue;
        if (!v || v.length !== DIM) continue;
        let ok = true;
        for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) { ok = false; break; }
        if (ok) m.add(letter, v);
      }
      return m;
    }
  }

  return { canonicalize, dist, Model };
})();
