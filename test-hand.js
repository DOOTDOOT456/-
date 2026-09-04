/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Synthetic hand builder + pose cases shared by the classifier and k-NN
   tests. Builds 21-point landmark arrays from joint angles and per-finger
   tip z-offsets (negative = toward the camera). Image coords, y grows
   downward, fingertips point up. Run inside node only. */
"use strict";

// MediaPipe hand landmark indices (also defined in asl.js).
const L = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

const SEGS = [0.13, 0.11, 0.10];

const Z = (x, y, z = 0) => [x, y, z];

/* Measure the PIP angle a given requested {base, pip, dip} produces, so
   requests can be tuned until they match the intended angles. */
function geometry(cfg) {
  const mcp = [0, 0];
  const b = cfg.base * Math.PI / 180;
  const pip = [mcp[0] + SEGS[0] * Math.cos(b), mcp[1] + SEGS[0] * Math.sin(b)];
  const a2 = b - (180 - cfg.pip) * Math.PI / 180;
  const dip = [pip[0] + SEGS[1] * Math.cos(a2), pip[1] + SEGS[1] * Math.sin(a2)];
  const a3 = a2 - (180 - cfg.dip) * Math.PI / 180;
  const tip = [dip[0] + SEGS[2] * Math.cos(a3), dip[1] + SEGS[2] * Math.sin(a3)];
  const d1 = Math.atan2(mcp[1] - pip[1], mcp[0] - pip[0]);
  const d2 = Math.atan2(tip[1] - pip[1], tip[0] - pip[0]);
  let d = Math.abs(d2 - d1) * 180 / Math.PI;
  return Math.min(d, 360 - d);
}

function tuned(cfg) {
  let lo = 1, hi = 179, req = cfg.pip;
  for (let i = 0; i < 14; i++) {
    const meas = geometry({ ...cfg, pip: req });
    if (Math.abs(meas - cfg.pip) < 1.5) break;
    if (meas < cfg.pip) lo = req; else hi = req;
    req = (lo + hi) / 2;
  }
  return { ...cfg, pip: req };
}

function buildHand(fingers, thumb, mcpOverrides = {}, tipZ = {}, wristPos = [0.5, 1.0]) {
  const mcpMap = Object.assign(
    { index: [0.33, 0.70], middle: [0.50, 0.66], ring: [0.70, 0.70], pinky: [0.88, 0.76] },
    mcpOverrides
  );
  const baseIdx = { index: L.INDEX_MCP, middle: L.MIDDLE_MCP, ring: L.RING_MCP, pinky: L.PINKY_MCP };
  const pts = {};
  pts[L.WRIST] = Z(...wristPos);
  pts[L.THUMB_CMC] = Z(0.16, 0.90);
  pts[L.THUMB_MCP] = Z(...thumb.mcp);
  pts[L.THUMB_IP] = Z(...thumb.ip);
  pts[L.THUMB_TIP] = Z(...thumb.tip);
  for (const n of ["index", "middle", "ring", "pinky"]) {
    const cfg = tuned(Object.assign({ base: -90, pip: 175, dip: 170 }, fingers[n] || {}));
    const mcp = mcpMap[n];
    const b = cfg.base * Math.PI / 180;
    const pip = [mcp[0] + SEGS[0] * Math.cos(b), mcp[1] + SEGS[0] * Math.sin(b)];
    const a2 = b - (180 - cfg.pip) * Math.PI / 180; // curl toward the palm (y down)
    const dip = [pip[0] + SEGS[1] * Math.cos(a2), pip[1] + SEGS[1] * Math.sin(a2)];
    const a3 = a2 - (180 - cfg.dip) * Math.PI / 180;
    const tip = [dip[0] + SEGS[2] * Math.cos(a3), dip[1] + SEGS[2] * Math.sin(a3)];
    pts[baseIdx[n]] = Z(...mcp);
    pts[baseIdx[n] + 1] = Z(...pip);
    pts[baseIdx[n] + 2] = Z(...dip);
    pts[baseIdx[n] + 3] = Z(tip[0], tip[1], tipZ[n] || 0);
  }
  return Object.keys(pts).sort((a, b) => a - b).map((k) => pts[k]);
}

const STRAIGHT = { pip: 175, dip: 170 };
const CURLED = { pip: 60, dip: 55 };
const TIGHT = { pip: 50, dip: 45 };

const THUMBS = {
  outUp:   { mcp: [0.12, 0.74], ip: [0.06, 0.66], tip: [0.02, 0.56] },   // Y / G / K / H / C
  outSide: { mcp: [0.10, 0.74], ip: [-0.04, 0.74], tip: [-0.18, 0.72] }, // A / L
  folded:  { mcp: [0.10, 0.74], ip: [0.18, 0.68], tip: [0.34, 0.84] },   // B / D / I / U / V / W
  over:    { mcp: [0.16, 0.70], ip: [0.36, 0.66], tip: [0.55, 0.70] },   // S (over the fingertips)
};

/* Static rule-supported letter poses (name + expected letter + fresh hand). */
const POSE_CASES = [
  { name: "B", want: "B", hand: () => buildHand({ index: STRAIGHT, middle: STRAIGHT, ring: STRAIGHT, pinky: STRAIGHT }, THUMBS.folded) },
  { name: "A", want: "A", hand: () => buildHand({ index: CURLED, middle: CURLED, ring: CURLED, pinky: CURLED }, THUMBS.outSide) },
  { name: "D", want: "D", hand: () => buildHand({ index: STRAIGHT, middle: CURLED, ring: CURLED, pinky: CURLED }, THUMBS.folded) },
  { name: "L", want: "L", hand: () => buildHand({ index: STRAIGHT, middle: CURLED, ring: CURLED, pinky: CURLED }, THUMBS.outSide) },
  { name: "G", want: "G", hand: () => buildHand({ index: STRAIGHT, middle: CURLED, ring: CURLED, pinky: CURLED }, THUMBS.outUp) },
  { name: "I", want: "I", hand: () => buildHand({ index: CURLED, middle: CURLED, ring: CURLED, pinky: STRAIGHT }, THUMBS.folded) },
  { name: "Y", want: "Y", hand: () => buildHand({ index: CURLED, middle: CURLED, ring: CURLED, pinky: STRAIGHT }, THUMBS.outUp) },
  { name: "F", want: "F", hand: () => buildHand({ index: { pip: 120, dip: 80 }, middle: STRAIGHT, ring: STRAIGHT, pinky: STRAIGHT },
    { mcp: [0.16, 0.70], ip: [0.16, 0.62], tip: [0.214, 0.500] }) }, // tip touches index tip
  { name: "O", want: "O", hand: () => buildHand({ index: TIGHT, middle: TIGHT, ring: TIGHT, pinky: TIGHT },
    { mcp: [0.16, 0.70], ip: [0.30, 0.64], tip: [0.346, 0.650] }) }, // tip touches index tip
  { name: "V", want: "V", hand: () => buildHand({ index: { base: -60, ...STRAIGHT }, middle: { base: -120, ...STRAIGHT }, ring: CURLED, pinky: CURLED }, THUMBS.folded) },
  { name: "U", want: "U", hand: () => buildHand({ index: { base: -80, ...STRAIGHT }, middle: { base: -100, ...STRAIGHT }, ring: CURLED, pinky: CURLED }, THUMBS.folded) },
  { name: "W", want: "W", hand: () => buildHand({ index: STRAIGHT, middle: STRAIGHT, ring: STRAIGHT, pinky: CURLED }, THUMBS.folded) },
  { name: "K", want: "K", hand: () => buildHand({ index: STRAIGHT, middle: STRAIGHT, ring: CURLED, pinky: CURLED },
    { mcp: [0.20, 0.72], ip: [0.30, 0.66], tip: [0.40, 0.60] }) },
  { name: "H", want: "H", hand: () => buildHand({ index: STRAIGHT, middle: STRAIGHT, ring: CURLED, pinky: CURLED }, THUMBS.outUp) },
  { name: "R", want: "R", hand: () => buildHand({ index: { base: -120, ...STRAIGHT }, middle: { base: -60, ...STRAIGHT }, ring: CURLED, pinky: CURLED },
    { mcp: [0.30, 0.74], ip: [0.40, 0.80], tip: [0.50, 0.86] }, // folded over the palm center
    { index: [0.52, 0.70], middle: [0.28, 0.66] },
    {}, [0.28, 1.0]) }, // wrist directly below middle MCP -> vertical hand axis
  { name: "X", want: "X", hand: () => buildHand({ index: { pip: 100, dip: 170 }, middle: CURLED, ring: CURLED, pinky: CURLED },
    THUMBS.folded, {}, { index: -0.12 }) }, // index hooked toward camera
  { name: "S", want: "S", hand: () => buildHand({ index: TIGHT, middle: TIGHT, ring: TIGHT, pinky: TIGHT }, THUMBS.over) },
  { name: "E", want: "E", hand: () => buildHand({ index: TIGHT, middle: TIGHT, ring: TIGHT, pinky: TIGHT }, THUMBS.folded) },
  { name: "T", want: "T", hand: () => buildHand({ index: CURLED, middle: CURLED, ring: CURLED, pinky: CURLED },
    { mcp: [0.20, 0.72], ip: [0.30, 0.66], tip: [0.40, 0.56] }) }, // thumb between index + middle
  { name: "C", want: "C", hand: () => buildHand({ index: { pip: 130, dip: 120 }, middle: { pip: 130, dip: 120 }, ring: { pip: 130, dip: 120 }, pinky: { pip: 130, dip: 120 } },
    { mcp: [0.18, 0.72], ip: [0.10, 0.62], tip: [0.02, 0.44] }) },
];

// ---- noise helpers -------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* Rotate a whole hand rigidly by deg degrees around the origin.
   Used to prove analyze() normalizes hand orientation: a rotated
   hand must classify exactly like the upright one. */
function rotateHand(lm, deg) {
  const r = deg * Math.PI / 180;
  const cos = Math.cos(r), sin = Math.sin(r);
  return lm.map((p) => [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos, p[2]]);
}

/* Return a fresh copy of lm with zero-mean gaussian noise (sigma, in
   normalized-coordinate units) added to every x/y/z. Deterministic per seed. */
function jitterHand(lm, sigma, seed) {
  const rng = mulberry32(seed);
  return lm.map((p) => [
    p[0] + gauss(rng) * sigma,
    p[1] + gauss(rng) * sigma,
    p[2] + gauss(rng) * sigma,
  ]);
}

module.exports = { L, SEGS, geometry, tuned, buildHand, STRAIGHT, CURLED, TIGHT, THUMBS, POSE_CASES, jitterHand, rotateHand };
