/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Builds signtype.html — the whole app as ONE self-contained HTML file:
   CSS and every local script inlined, and (when seed-db.json exists) the
   trained learning DB embedded so anyone who opens the file starts with
   the pre-trained hand model. No server, no CDN needed except MediaPipe
   (loaded from jsdelivr at runtime).

   Usage:
     node tools/build-single.mjs
   Output: signtype.html

   Owner workflow: open the app -> ✋ Train the shared hand model ->
   sign through the alphabet -> ⬇ Export learned DB -> save the JSON as
   seed-db.json in the project root -> run this script -> ship
   signtype.html (+ seed-db.json for the multi-file version). */
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

// 2. embed the trained seed DB (if present) before app.js runs
if (fs.existsSync(path.join(root, "seed-db.json"))) {
  const seed = fs.readFileSync(path.join(root, "seed-db.json"), "utf8");
  // sanity: must be parseable JSON with samples
  const parsed = JSON.parse(seed);
  if (!parsed || !Array.isArray(parsed.samples)) {
    console.error("seed-db.json is not a valid SignType learning DB — fix or remove it.");
    process.exit(1);
  }
  html = html.replace(
    '<script src="app.js"></script>',
    `<script>window.SEED_DB = ${seed};</script>\n<script src="app.js"></script>`
  );
  console.log("Embedded seed DB:", parsed.samples.length, "samples");
} else {
  console.log("No seed-db.json found — building without a pre-trained seed.");
}

// 3. inline every local script (CDN scripts stay as <script src>)
html = html.replace(/<script src="([^"]+\.js)"><\/script>/g, (m, src) => {
  if (/^https?:/.test(src)) return m;
  return "<script>\n" + read(src) + "\n</script>";
});

const out = path.join(root, "signtype.html");
fs.writeFileSync(out, html);
console.log("Built", out, "-", (html.length / 1024).toFixed(0), "KB");