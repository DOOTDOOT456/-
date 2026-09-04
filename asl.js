/* Copyright © 2026 [Your Full Name]. All rights reserved.

   ASL letter recognition from MediaPipe hand landmarks.
   Pure functions — no DOM access, safe to unit-test.
   Landmarks: 21 points, each {x, y, z} with x,y normalized to [0,1]. */
"use strict";

const ASL = (function () {

  const L = {
    WRIST: 0,
    THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
    INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
    MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
    RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
    PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
  };

  const FINGERS = {
    index:  { mcp: L.INDEX_MCP,  pip: L.INDEX_PIP,  dip: L.INDEX_DIP,  tip: L.INDEX_TIP },
    middle: { mcp: L.MIDDLE_MCP, pip: L.MIDDLE_PIP, dip: L.MIDDLE_DIP, tip: L.MIDDLE_TIP },
    ring:   { mcp: L.RING_MCP,   pip: L.RING_PIP,   dip: L.RING_DIP,   tip: L.RING_TIP },
    pinky:  { mcp: L.PINKY_MCP,  pip: L.PINKY_PIP,  dip: L.PINKY_DIP,  tip: L.PINKY_TIP },
  };

  // ---- vector helpers ----
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
  function unit(a, b) { const d = sub(a, b); const n = norm(d) || 1; return [d[0] / n, d[1] / n, d[2] / n]; }
  function angleDeg(a, b, c) {
    const v1 = sub(a, b), v2 = sub(c, b);
    const n1 = norm(v1), n2 = norm(v2);
    if (n1 < 1e-6 || n2 < 1e-6) return 180;
    let cos = dot(v1, v2) / (n1 * n2);
    cos = Math.max(-1, Math.min(1, cos));
    return Math.acos(cos) * 180 / Math.PI;
  }

  // A finger counts as extended only if BOTH knuckles are nearly straight.
  const EXT_PIP = 150, EXT_DIP = 145;
  const CURL_PIP = 115, CURL_DIP = 115;

  /* Rotate the landmark cloud so the wrist->middle-MCP axis points up on
     screen. Features that compare "above/below" or "left/right" (thumb
     above the knuckle row, outside the finger span) are then relative to
     the hand itself, so tilted hands, upside-down angles, and either hand
     behave consistently. Pure rotation: distances, angles and depth are
     unchanged, so every other feature keeps its exact semantics. */
  function normalizePose(raw) {
    const lm = raw.map((p) => (Array.isArray(p) ? p : [p.x, p.y, p.z]));
    const wrist = lm[L.WRIST], mid = lm[L.MIDDLE_MCP];
    const dx = mid[0] - wrist[0], dy = mid[1] - wrist[1];
    const s = Math.hypot(dx, dy);
    if (!(s > 1e-6)) return lm;
    const phi = -Math.atan2(dy, dx) - Math.PI / 2;
    const cos = Math.cos(phi), sin = Math.sin(phi);
    return lm.map((p) => [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos, p[2]]);
  }

  /* Compute classification features from a flat landmark array
     (each entry [x, y, z] or {x, y, z}). */
  function analyze(raw) {
    const lm = normalizePose(raw);
    const wrist = lm[L.WRIST];
    const handSize = dist(lm[L.MIDDLE_MCP], wrist) || 1e-6; // scale reference

    const pip = {}, dip = {}, ext = {}, curl = {};
    for (const name of Object.keys(FINGERS)) {
      const f = FINGERS[name];
      pip[name] = angleDeg(lm[f.mcp], lm[f.pip], lm[f.tip]);
      dip[name] = angleDeg(lm[f.pip], lm[f.dip], lm[f.tip]);
      ext[name] = pip[name] > EXT_PIP && dip[name] > EXT_DIP;
      curl[name] = pip[name] < CURL_PIP && dip[name] < CURL_DIP;
    }

    // Thumb: "out" = the tip sticks out from the palm — either clearly above
    // the finger-MCP row (Y/L/G/K/H/C) or outside the finger span (A sideways).
    // A folded/crossed thumb (B/D/E/I/O/S/U/V/W) hangs over the palm instead.
    const thumbTip = lm[L.THUMB_TIP];
    const aboveRow = thumbTip[1] < lm[L.INDEX_MCP][1] - 0.015;
    const outsideSpan =
      thumbTip[0] < lm[L.INDEX_MCP][0] - 0.05 ||
      thumbTip[0] > lm[L.PINKY_MCP][0] + 0.05;
    const thumbOut = aboveRow || outsideSpan;
    const indexTip = lm[L.INDEX_TIP];
    const middleTip = lm[L.MIDDLE_TIP];
    const ringTip = lm[L.RING_TIP];

    const thumbTipIndexTip = dist(thumbTip, indexTip) / handSize;
    const thumbTipMiddleTip = dist(thumbTip, middleTip) / handSize;
    const thumbTipRingTip = dist(thumbTip, ringTip) / handSize;

    // Where is the thumb tip among the curled fingers? (A / T / M family)
    const dTi = dist(thumbTip, lm[L.INDEX_MCP]) / handSize;
    const dTm = dist(thumbTip, lm[L.MIDDLE_MCP]) / handSize;
    const dTr = dist(thumbTip, lm[L.RING_MCP]) / handSize;
    let thumbPos = "a"; // alongside the index
    if (dTi < 0.5 && dTm < 0.45) thumbPos = "t";          // between index + middle
    else if (dTm < 0.45 && dTr < 0.45) thumbPos = "m";    // under middle + ring

    // X: index hooked toward the camera while the rest stay tucked
    // (the hook shows up in depth — z grows negative toward the camera)
    const indexForwardZ = lm[L.INDEX_TIP][2] < lm[L.MIDDLE_TIP][2] - 0.06;
    const indexProtrudes = indexForwardZ ||
      dist(lm[L.INDEX_TIP], wrist) > dist(lm[L.MIDDLE_TIP], wrist) + 0.3 * handSize;

    // S vs E: S wraps the thumb OVER the fingers (near the ring side),
    // E keeps the thumb on the index side.
    const thumbCloserToRing = thumbTipRingTip < thumbTipIndexTip;

    // R: index + middle crossed (MCP order flips vs tip order)
    const mcpOrder = lm[L.INDEX_MCP][0] < lm[L.MIDDLE_MCP][0];
    const tipOrder = indexTip[0] < middleTip[0];
    const crossed = mcpOrder !== tipOrder && dist(indexTip, middleTip) / handSize < 0.4;

    // V vs U: spread fingers
    const spread = dist(indexTip, middleTip) / handSize > 0.4;

    // K vs H: K's thumb sits right at the base between index + middle
    const midMcp = [
      (lm[L.INDEX_MCP][0] + lm[L.MIDDLE_MCP][0]) / 2,
      (lm[L.INDEX_MCP][1] + lm[L.MIDDLE_MCP][1]) / 2,
      (lm[L.INDEX_MCP][2] + lm[L.MIDDLE_MCP][2]) / 2,
    ];
    const kThumb = dist(thumbTip, midMcp) / handSize < 0.5;

    // G vs L: angle between the index and thumb directions
    const indexDir = unit(lm[L.INDEX_TIP], lm[L.INDEX_MCP]);
    const thumbDir = unit(lm[L.THUMB_TIP], lm[L.THUMB_MCP]);
    const thumbIndexAngle =
      Math.acos(Math.max(-1, Math.min(1, dot(indexDir, thumbDir)))) * 180 / Math.PI;

    const pattern = [ext.index, ext.middle, ext.ring, ext.pinky].map(Number).join("");
    const minPip = Math.min(pip.index, pip.middle, pip.ring, pip.pinky);
    const maxPip = Math.max(pip.index, pip.middle, pip.ring, pip.pinky);

    return {
      pip, dip, ext, curl,
      thumbOut, thumbPos, indexProtrudes, thumbCloserToRing,
      thumbTipIndexTip, thumbTipMiddleTip, thumbTipRingTip,
      crossed, spread, kThumb, thumbIndexAngle,
      pattern, minPip, maxPip, handSize,
    };
  }

  /* Map features to a letter. Returns null when nothing confident matches.
     " " (space) is intentionally NOT returned here — the app maps the
     open-hand pose to space itself. */
  function classify(f) {
    const p = f.pattern;
    // A real fist: no extended fingers AND the knuckles are tightly bent
    // (a half-open curved hand is C, not a fist).
    const fist = p === "0000" && f.minPip < 115;
    const touchingIndex = f.thumbTipIndexTip < 0.32;

    // Open hand with all fingers straight: B when the thumb is tucked,
    // otherwise the open palm ("5" / space) handled by the app.
    if (p === "1111") return f.thumbOut ? null : "B";

    // Thumb tip touching the index tip: F (fingers up) or O (tight fist)
    if (touchingIndex) {
      if (!f.ext.index && f.ext.middle && f.ext.ring && f.ext.pinky) return "F";
      if (fist) return "O";
    }

    // Tight fist
    if (fist) {
      if (f.thumbOut) {
        if (f.thumbPos === "t") return "T";
        if (f.thumbPos === "m") return "M";
        return "A";
      }
      if (f.indexProtrudes) return "X";
      if (f.thumbCloserToRing) return "S";
      return "E";
    }

    // C: half-open curved hand, thumb out, no finger fully extended
    if (!f.ext.index && !f.ext.middle && !f.ext.ring && !f.ext.pinky &&
        f.thumbOut && f.minPip > 100 && f.maxPip < 160 && !touchingIndex) {
      return "C";
    }

    // Only index extended
    if (p === "1000") {
      if (f.thumbOut) return f.thumbIndexAngle < 50 ? "G" : "L";
      return "D";
    }

    // Index + middle extended
    if (p === "1100") {
      if (f.thumbOut) return f.kThumb ? "K" : "H";
      if (f.crossed) return "R";
      if (f.spread) return "V";
      return "U";
    }

    // Index + middle + ring
    if (p === "1110") return "W";

    // Pinky only
    if (p === "0001") return f.thumbOut ? "Y" : "I";

    return null;
  }

  /* Motion letters (J, Z): classify a recent path of index-tip positions
     [{x, y, t}] in normalized image coordinates. y grows downward.
     Returns "J", "Z" or null. Mirror-safe: horizontal directions are
     compared relatively, so both hands work. */
  function classifyMotion(pts) {
    if (!pts || pts.length < 14) return null;
    const start = pts[0], end = pts[pts.length - 1];
    const totalX = end.x - start.x, totalY = end.y - start.y;
    const totalMag = Math.hypot(totalX, totalY);
    if (totalMag < 0.15) return null;

    const n = pts.length;
    const s1 = pts[Math.floor(n / 3)], s2 = pts[Math.floor(2 * n / 3)];
    const d1 = [s1.x - start.x, s1.y - start.y];
    const d2 = [s2.x - s1.x, s2.y - s1.y];
    const d3 = [end.x - s2.x, end.y - s2.y];
    const m1 = Math.hypot(d1[0], d1[1]);
    const m2 = Math.hypot(d2[0], d2[1]);
    const m3 = Math.hypot(d3[0], d3[1]);
    if (m1 < 0.05 || m2 < 0.05 || m3 < 0.05) return null;

    // Z: horizontal stroke → downward diagonal back toward the start → horizontal stroke
    const z =
      d1[0] * d3[0] > 0 &&                              // first + last go the same way
      Math.abs(d1[0]) > 0.4 * m1 && Math.abs(d1[1]) < 0.7 * Math.abs(d1[0]) &&
      Math.abs(d3[0]) > 0.4 * m3 && Math.abs(d3[1]) < 0.7 * Math.abs(d3[0]) &&
      d2[1] > 0.2 * m2 && d2[0] * d1[0] < 0 &&          // middle drops diagonally back
      Math.abs(totalY) < 0.6 * totalMag;

    // J: downward stroke → any curve → upward hook
    const j =
      d1[1] > 0.3 * m1 && Math.abs(d1[0]) < 0.9 * m1 &&
      d3[1] < -0.25 * m3 && Math.abs(d3[0]) < 0.9 * m3 &&
      totalY > 0;

    if (z) return "Z";
    if (j) return "J";
    return null;
  }

  const LETTERS = {
    A: { desc: "Fist, thumb out to the side", emoji: "✊", rel: "solid" },
    B: { desc: "Four fingers up, thumb tucked", emoji: "✋", rel: "solid" },
    C: { desc: "Half-open curved hand", emoji: "", rel: "approx" },
    D: { desc: "Index up, others curled", emoji: "👆", rel: "solid" },
    E: { desc: "Fingers curled, thumb in front", emoji: "", rel: "approx" },
    F: { desc: "Thumb touches index tip, three fingers up", emoji: "🤏", rel: "approx" },
    G: { desc: "Index + thumb forward, others curled", emoji: "", rel: "approx" },
    H: { desc: "Index + middle up, thumb resting on them", emoji: "", rel: "approx" },
    I: { desc: "Pinky up, others curled", emoji: "", rel: "solid" },
    J: { desc: "Trace a J in the air with your index", emoji: "", rel: "motion" },
    K: { desc: "Index + middle up, thumb between them", emoji: "", rel: "approx" },
    L: { desc: "Index + thumb up in an L", emoji: "", rel: "solid" },
    M: { desc: "Fist, thumb tucked under middle + ring", emoji: "", rel: "approx" },
    N: { desc: "Fist, thumb under index + middle", emoji: "", rel: "solid" },
    O: { desc: "All fingertips touch the thumb tip", emoji: "👌", rel: "approx" },
    P: { desc: "Index + thumb point down", emoji: "", rel: "solid" },
    Q: { desc: "Like P with the thumb hanging lower", emoji: "", rel: "solid" },
    R: { desc: "Index + middle crossed", emoji: "🤞", rel: "approx" },
    S: { desc: "Fist, thumb over the fingers", emoji: "", rel: "approx" },
    T: { desc: "Fist, thumb between index + middle", emoji: "", rel: "approx" },
    U: { desc: "Index + middle up, side by side", emoji: "", rel: "approx" },
    V: { desc: "Index + middle up, spread apart", emoji: "✌️", rel: "solid" },
    W: { desc: "Index + middle + ring up", emoji: "", rel: "solid" },
    X: { desc: "Index hooked, others curled", emoji: "", rel: "approx" },
    Y: { desc: "Thumb + pinky out", emoji: "🤙", rel: "solid" },
    Z: { desc: "Trace a Z in the air with your index", emoji: "", rel: "motion" },
  };

  return { L, FINGERS, analyze, classify, classifyMotion, normalizePose, LETTERS };
})();