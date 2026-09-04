/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Builds signtype.html — the whole app as ONE self-contained HTML file:
   CSS and every local script inlined, and (when db.json / seed-db.json
   exists) the trained learning database embedded so anyone who opens the
   file starts with the pre-trained hand model. No server, no CDN needed
   except MediaPipe (loaded from jsdelivr at runtime).

   Usage:
     node tools/build-single.mjs
   Output: signtype.html

   Owner workflow: open the app -> 🧠 Build my sign database -> sign
   through the alphabet -> ⬇ Export db.json -> save the JSON as db.json
   in the project root -> run this script -> ship signtype.html
   (or keep db.json next to index.html for the multi-file version). */
"use strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
function read(p) { return fs.readFileSync(path.join(root, p), "utf8"); }

let html = read("index.html");

// 1. inline the stylesheet
html = html.replace(
  '<link rel="stylesheet" href="style.css">',
  "<style>\n" + read("style.css") + "\n</style>"
);

// 2. embed the trained database (db.json preferred, seed-db.json fallback)
//    before app.js runs
const SEED_CANDIDATES = ["db.json", "seed-db.json"];
let seedFile = null;
for (const f of SEED_CANDIDATES) {
  if (fs.existsSync(path.join(root, f))) { seedFile = f; break; }
}
if (seedFile) {
  const seed = fs.readFileSync(path.join(root, seedFile), "utf8");
  // sanity: must be parseable JSON with samples
  const parsed = JSON.parse(seed);
  if (!parsed || !Array.isArray(parsed.samples)) {
    console.error(seedFile + " is not a valid SignType learning DB — fix or remove it.");
    process.exit(1);
  }
  html = html.replace(
    '<script src="app.js"></script>',
    `<script>window.SEED_DB = ${seed};</script>\n<script src="app.js"></script>`
  );
  console.log("Embedded", seedFile, ":", parsed.samples.length, "samples");
} else {
  console.log("No db.json / seed-db.json found — building without a pre-trained database.");
}

// 3. inline every local script (CDN scripts stay as <script src>)
html = html.replace(/<script src="([^"]+\.js)"><\/script>/g, (m, src) => {
  if (/^https?:/.test(src)) return m;
  return "<script>\n" + read(src) + "\n</script>";
});

const out = path.join(root, "signtype.html");
fs.writeFileSync(out, html);
console.log("Built", out, "-", (html.length / 1024).toFixed(0), "KB");