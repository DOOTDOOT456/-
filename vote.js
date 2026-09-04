/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Sliding-window majority voter for per-frame recognition results.
   Pure logic — no DOM access, safe to unit-test.

   A single misread frame used to reset the whole "hold steady" timer,
   so flicker between look-alike letters (A/S/E, H/U, V/K...) made
   typing feel broken. The voter smooths the per-frame letter stream:
   a letter only becomes active once it wins a majority of the last N
   frames, and one glitchy frame can no longer break a stable hold. */
"use strict";

const VOTE = (function () {

  class SlidingVote {
    /* size: how many recent frames to consider.
       minShare: fraction of the window a letter must win (0..1). */
    constructor(size = 10, minShare = 0.6) {
      this.size = size;
      this.minVotes = Math.ceil(minShare * size);
      this.win = [];
    }

    /* Record one frame's letter (null = no confident match) and return
       the current majority letter, or null when no letter holds the
       required share. Nulls decay the window without vetoing it, so a
       few unrecognized frames can't kill a valid hold. */
    push(v) {
      this.win.push(v);
      if (this.win.length > this.size) this.win.shift();
      return this.majority();
    }

    majority() {
      const counts = {};
      for (const v of this.win) {
        if (v === null || v === undefined) continue;
        counts[v] = (counts[v] || 0) + 1;
      }
      let best = null, bestN = 0;
      for (const k of Object.keys(counts)) {
        if (counts[k] > bestN) { bestN = counts[k]; best = k; }
      }
      return bestN >= this.minVotes ? best : null;
    }

    clear() { this.win.length = 0; }
  }

  return { SlidingVote };
})();