# SignType — ASL letter practice

A browser app that detects American Sign Language hand shapes with your
webcam and auto-types letters so you can practice fingerspelling.

## Features

- **Camera + hand tracking** — the modern MediaPipe Hand Landmarker
  task (MediaPipe Tasks Vision, GPU-accelerated, VIDEO running mode)
  draws a live skeleton overlay on your webcam feed (mirrored). It
  tracks the hand between frames with a lightweight matcher and only
  re-runs palm detection when tracking fails, so it holds onto your
  hand far more reliably under motion, tilt and partial occlusion than
  the old deprecated `@mediapipe/hands` solution. No camera? Click the
  on-screen keyboard instead.
- **Learned recognition engine** — the primary classifier is a
  weighted k-NN over real hand samples (`knn.js` + `db.js`), covering
  all 24 static letters **A–Y, N/P/Q included**. The shipped seed
  (`seed-db.json`) is built from 4064 real MediaPipe hand landmarks
  (Siruyy/asl-static-landmarks-v1, CC-BY-4.0), canonicalized (position,
  size, tilt and left/right hand normalized away), self-cleaned by
  leave-one-out k-NN, and measures **83.8% leave-one-out accuracy** on
  real hands — **≥94% precise at confidence ≥0.75**, which is when it
  overrides the geometric rules. The tested rule classifier (`asl.js`)
  is the default otherwise and handles open-palm space plus **J**/**Z**
  motion tracing. (The old neural net was found to be trained on an
  undocumented preprocessing that never worked on real inputs — it
  collapsed to a few classes on everything tested — so it was removed
  from the decision path; files remain for provenance.)
- **Anti-flicker voting** — per-frame guesses are smoothed by a
  sliding-window majority vote, so one glitchy frame can't reset a hold
  or type the wrong letter; hand orientation is normalized first, so
  tilted hands and either hand classify like upright ones.
- **Learns from every signer (no training needed)** — confident holds
  are stored in a small in-browser database (`db.js`, IndexedDB), grouped
  per letter, deduped and capped per letter. The collection *is* the
  classifier, so recognition gets better the more someone uses the app —
  for their own hand, their own way.
- **Pre-trained seed** — a real-hand seed ships with the app already
  (no training needed by anyone). The ✋ *Train the shared hand model*
  tool lets the owner add their own hand on top: sign letters, click
  **⬇ Export learned DB**, ship the JSON as `seed-db.json`. The app ships
  as one self-contained file too: `node tools/build-single.mjs` builds
  `signtype.html` with all code and the seed DB inlined.
- **Three practice modes** — Free type, Alphabet drill with progress
  bar, and editable Phrase drill.

## Run it

```bash
node serve.js 8000        # or python3 -m http.server 8000
# then open http://localhost:8000 — or just double-click index.html
```

Webcam access needs a secure context: use `http://localhost`, not a raw
`file://` double-click in some browsers.

## Tests

```bash
node classifier.test.js   # rule-based classifier (incl. tilt invariance)
node knn.test.js          # k-NN canonicalization + accuracy
node vote.test.js         # temporal smoothing voter
node db.test.js           # learning library: dedupe, caps, seed export/import
node net.test.js          # archived neural model artifact integrity (provenance)
```

## Seed provenance / retraining

- `seed-db.json` is built by `tools/build-seed.mjs` from **4064 real
  MediaPipe hand landmarks** across all 26 letters —
  [Siruyy/asl-static-landmarks-v1](https://huggingface.co/datasets/Siruyy/asl-static-landmarks-v1)
  (CC-BY-4.0). The script un-standardizes the raw landmark columns,
  canonicalizes them (`knn.js`), collapses near-duplicates, drops
  label-noise samples via leave-one-out k-NN disagreement, and reports
  the per-letter accuracy of what ships.
  (The npy files are downloaded into `tools/data/siruyy/` — gitignored,
  ~5 MB.)
- `tools/audit-db.mjs` scores any exported learning DB against reference
  poses and cluster stats — use it to vet training before shipping a
  seed.
- `tools/build-single.mjs` inlines everything (and the seed) into one
  `signtype.html`.
- **Why the neural net was archived:** `asl-net.json` (AmimulBmeIU's
  ASL Alphabet Classifier) loads and its weights are byte-exact vs the
  original, but its training preprocessing is undocumented. Feeding it
  real landmarks — and the app's own synthetic poses — collapses it to a
  few classes (13.5% label agreement on real data, 2/20 on synthetic,
  across every plausible normalization). It never worked end-to-end, so
  it was removed from the decision path; `net.js`, `asl-net.json`,
  `tools/export-net.mjs`, `tools/parity-tfjs.mjs` and `net.test.js` stay
  for provenance.

## Third-party notices

- `seed-db.json` contains landmark samples from
  **Siruyy/asl-static-landmarks-v1**
  (https://huggingface.co/datasets/Siruyy/asl-static-landmarks-v1), ©
  its author, CC-BY-4.0.
- `asl-net.json` (archived, not loaded by the app) contains weights from
  the **ASL Alphabet Classifier**
  (https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier), © its
  author, MIT License.
- MediaPipe Hand Landmarker task (`@mediapipe/tasks-vision`, bundled
  WASM runtime) — Google, Apache-2.0 (loaded from cdn.jsdelivr.net);
  the `hand_landmarker.task` model is served by Google's model storage.
  Landmark drawing is done in-app (no external drawing utils).

## License

Copyright © 2026 [Your Full Name]. All rights reserved — see
[LICENSE](LICENSE). This project is not open source; copying or reusing
any part of it requires written permission from the owner. (The seed's
CC-BY-4.0 samples and the archived net's MIT weights keep their own
licenses as noted above.)
