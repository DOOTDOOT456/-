# SignType — ASL letter practice

A browser app that detects American Sign Language hand shapes with your
webcam and auto-types letters so you can practice fingerspelling.

## Features

- **Camera + hand tracking** — MediaPipe Hands draws a live skeleton
  overlay on your webcam feed (mirrored). No camera? Click the on-screen
  keyboard instead.
- **Neural recognition engine** — a small neural network (shipped as
  `asl-net.json`, executed by the dependency-free `net.js`) classifies
  the 21 hand landmarks into static letters **A–Y, N/P/Q included**. The
  weights come from the *ASL Alphabet Classifier* by AmimulBmeIU
  (MIT license, 94.15% reported validation accuracy) — see
  [Third-party notices](#third-party-notices). The tested rule-based
  classifier (`asl.js`) stays in charge whenever the net is unsure or
  disagrees, so the neural engine can only add letters, never regress the
  rules. Open palm types a space; **J** and **Z** are traced in the air.
- **Anti-flicker voting** — per-frame guesses are smoothed by a
  sliding-window majority vote, so one glitchy frame can't reset a hold
  or type the wrong letter; hand orientation is normalized first, so
  tilted hands and either hand classify like upright ones.
- **✍️ My signs** — record 3 samples per letter with your own hand and
  the app classifies with a weighted k-NN model instead (covers any
  letter you teach it). The model auto-saves in your browser
  (localStorage).
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
node net.test.js          # neural model structure + forward-pass integrity
```

## Retraining / model provenance

- `tools/export-net.mjs` re-exports `asl-net.json` from the original
  TensorFlow.js files in `tools/model/` (downloadable from
  https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier).
- `tools/parity-tfjs.mjs` loads the real TF.js model (Node + the wasm
  backend, `npm i` in `tools/`) and proves the sliced weights reproduce
  it exactly (max output diff < 1e-6). `net.test.js` guards the shipped
  artifact's integrity.
- Honest limits: this box had no GPU and no way to run MediaPipe on
  images headlessly, so the neural engine was verified for weight
  fidelity and forward-pass correctness — but not end-to-end on a live
  webcam. The rules-first gating means that if the net misreads your
  hand, you get the previous rule-based reading rather than a new error.

## Third-party notices

- `asl-net.json` contains weights from the **ASL Alphabet Classifier**
  (https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier), © its
  author, MIT License. The model maps 21 MediaPipe hand landmarks to the
  24 static ASL letters A–Y (no J/Z).
- MediaPipe Hands, camera utils, drawing utils — Google, Apache-2.0
  (loaded from cdn.jsdelivr.net).

## License

Copyright © 2026 [Your Full Name]. All rights reserved — see
[LICENSE](LICENSE). This project is not open source; copying or reusing
any part of it requires written permission from the owner. (The
`asl-net.json` weights keep their MIT license as noted above.)
