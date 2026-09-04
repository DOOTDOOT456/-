/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Sliding-vote sanity checks — run with: node vote.test.js
   Verifies the temporal majority voter used to smooth per-frame
   recognition: a letter needs a majority of the window to activate,
   single glitches can't break a hold, sustained switches flip the
   vote, and nulls decay it. */
"use strict";

const fs = require("fs");
const VOTE = new Function(fs.readFileSync("vote.js", "utf8") + "\nreturn VOTE;")();

let pass = 0, fail = 0;
function check(name, cond) {
  cond ? pass++ : fail++;
  console.log(`${cond ? "PASS" : "FAIL"} ${name}`);
}

// empty / null-only window never produces a letter
{
  const v = new VOTE.SlidingVote(10, 0.6);
  check("null window -> null", v.push(null) === null);
  check("nulls never vote", v.push(null) === null && v.push(null) === null);
}

// majority needs 6 of 10 frames
{
  const v = new VOTE.SlidingVote(10, 0.6);
  let r = null;
  for (let i = 0; i < 5; i++) r = v.push("A");   // 5 A's + 1 null
  check("5 of 10 not enough", r === null);
  check("6 of 10 activates", v.push("A") === "A");
}

// a single glitchy frame cannot break an active majority
{
  const v = new VOTE.SlidingVote(10, 0.6);
  for (let i = 0; i < 6; i++) v.push("A");
  check("glitch ignored", v.push("B") === "A");
  check("still holding after 2nd glitch", v.push("B") === "A");
}

// a sustained switch eventually flips the vote
{
  const v = new VOTE.SlidingVote(10, 0.6);
  for (let i = 0; i < 6; i++) v.push("A");
  let r = "A";
  for (let i = 0; i < 4; i++) r = v.push("B");   // A 6, B 4 -> A still wins
  check("minority switch does not flip yet", r === "A");
  r = v.push("B");                               // A 5, B 5 -> no majority
  check("tie -> no majority", r === null);
  for (let i = 0; i < 2; i++) r = v.push("B");   // B 6, A 5 -> flips
  check("majority switch flips", r === "B");
}

// nulls decay an old letter instead of letting it linger forever
{
  const v = new VOTE.SlidingVote(10, 0.6);
  for (let i = 0; i < 6; i++) v.push("Z");
  let r = "Z";
  for (let i = 0; i < 4; i++) r = v.push(null);   // 6 Z + 4 nulls -> Z still wins
  check("Z holds through a few bad frames", r === "Z");
  check("Z fully decayed", v.push(null) === null); // 5th null tips the window
}

// clear() empties the window
{
  const v = new VOTE.SlidingVote(10, 0.6);
  for (let i = 0; i < 6; i++) v.push("Z");
  v.clear();
  check("clear empties window", v.push("Z") === null);
}

// space (" ") participates like any letter
{
  const v = new VOTE.SlidingVote(10, 0.6);
  for (let i = 0; i < 5; i++) v.push(" ");
  check("space needs majority too", v.push(" ") === " " && v.push(" ") === " ");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);