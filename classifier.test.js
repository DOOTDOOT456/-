/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Quick classifier sanity checks — run with: node classifier.test.js
   Builds synthetic hand landmark arrays from joint angles and checks
   that analyze()/classify()/classifyMotion() return the expected letters. */
"use strict";

const fs = require("fs");
const { buildHand, STRAIGHT, CURLED, TIGHT, THUMBS, POSE_CASES, rotateHand } = require("./test-hand.js");

// Load the browser-style script and grab its ASL global.
const ASL = new Function(fs.readFileSync("asl.js", "utf8") + "\nreturn ASL;")();

let pass = 0, fail = 0;
for (const { name, want, hand } of POSE_CASES) {
  const feats = ASL.analyze(hand());
  const got = ASL.classify(feats);
  const ok = got === want;
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: expected ${want}, got ${got}` +
    (ok ? "" : `  [pattern ${feats.pattern}, thumbOut ${feats.thumbOut}]`));
}

// open hand (thumb out) should NOT classify as a letter (app maps it to space)
{
  const lm = buildHand({ index: STRAIGHT, middle: STRAIGHT, ring: STRAIGHT, pinky: STRAIGHT }, THUMBS.outSide);
  const got = ASL.classify(ASL.analyze(lm));
  const ok = got === null;
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"} open-hand(space): expected null, got ${got}`);
}

// motion letters
function interp(a, b, steps) {
  const out = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    out.push({ x: a[0] + (b[0] - a[0]) * t, y: a[1] + (b[1] - a[1]) * t, t: i });
  }
  return out;
}
const zPath = [...interp([0.30, 0.50], [0.60, 0.50], 5),
               ...interp([0.60, 0.50], [0.35, 0.70], 5),
               ...interp([0.35, 0.70], [0.65, 0.70], 5)];
const jPath = [...interp([0.40, 0.30], [0.45, 0.60], 5),
               ...interp([0.45, 0.60], [0.42, 0.80], 5),
               ...interp([0.42, 0.80], [0.32, 0.60], 5)];
const squiggle = [...interp([0.30, 0.30], [0.60, 0.60], 5),
                  ...interp([0.60, 0.60], [0.35, 0.35], 5),
                  ...interp([0.35, 0.35], [0.55, 0.55], 5)];

for (const [name, path, want] of [
  ["Z", zPath, "Z"],
  ["J", jPath, "J"],
  ["squiggle", squiggle, null],
]) {
  const got = ASL.classifyMotion(path);
  const ok = got === want;
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"} motion-${name}: expected ${want}, got ${got}`);
}

// Tilt invariance: the same pose rotated in the frame must classify the
// same way (analyze() normalizes hand orientation before extracting
// features, so tilted hands and left hands behave like upright ones).
for (const deg of [35, -25]) {
  for (const { name, want, hand } of POSE_CASES) {
    const got = ASL.classify(ASL.analyze(rotateHand(hand(), deg)));
    const ok = got === want;
    ok ? pass++ : fail++;
    if (!ok) console.log(`FAIL ${name} rotated ${deg}°: expected ${want}, got ${got}`);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
