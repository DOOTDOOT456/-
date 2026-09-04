/* Copyright © 2026 [Your Full Name]. All rights reserved.

   SignType — app glue: webcam, hand tracking, typing logic, UI.
   Recognition engines:
     - learned (db.js + knn.js): the primary engine. A database of real
       sign samples (a shipped seed trained on real hands, plus samples
       collected automatically from every confident hold, grouped per
       letter) classified with weighted k-NN. Calibrated on real data:
       ~84% leave-one-out accuracy overall, and >=94% precise at
       confidence >= 0.75 — so it can safely override the rules.
     - rules (asl.js): the tested geometric classifier — the default for
       letters the learned engine hasn't seen, open-hand space, and J/Z
       motion tracing.
     - The old neural engine (net.js + asl-net.json) was removed from
       the decision path: its training preprocessing is undocumented and
       it provably never worked on real inputs (it collapses to a few
       classes on every input tested). Files remain for provenance. */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
  const statusEl = $("status"), featuresEl = $("features");
  const currentLetterEl = $("current-letter"), currentNameEl = $("current-letter-name");
  const handHint = $("hand-hint"), cameraError = $("camera-error");
  const keyboardEl = $("keyboard"), guideGridEl = $("guide-grid");
  const typedBox = $("typed-box"), typedSub = $("typed-sub");
  const drillEl = $("drill"), drillTarget = $("drill-target"), drillBar = $("drill-bar");
  const phraseRow = $("phrase-row"), phraseInput = $("phrase-input"), phraseStart = $("phrase-start");
  const copyBtn = $("copy-btn"), clearBtn = $("clear-btn"), soundToggle = $("sound-toggle");
  const seedTrain = $("seed-train"), seedSummary = $("seed-summary");
  const seedBig = $("seed-big"), seedName = $("seed-name"), seedDots = $("seed-dots");
  const seedPrev = $("seed-prev"), seedNext = $("seed-next");
  const seedExport = $("seed-export"), seedImport = $("seed-import");
  const seedImportFile = $("seed-import-file"), seedClear = $("seed-clear");

  // ---------------- constants ----------------
  const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const LEGACY_MODEL_KEY = "signtype-knn-model-v1"; // My-signs model, migrated once
  const CONF_MIN = 0.55;        // learned k-NN confidence gate (0..1)
  const NN_MAX = 1.2;           // learned k-NN nearest-neighbor distance gate
  const LEARNED_OVERRIDE = 0.75; // learned conf above this can override the rules
  // Shipped database the client loads so every visitor starts pre-trained.
  // "db.json" is the file you build with the export button; "seed-db.json"
  // is kept as a fallback for existing deployments.
  const SEED_CANDIDATES = ["db.json", "seed-db.json"];
  const SEED_GOAL = 5;          // seed trainer: dots shown per letter (soft target)
  const MAX_DOTS = 8;           // cap dots rendered per letter

  // ---------------- state ----------------
  const state = {
    mode: "free",
    typed: "",
    drillIdx: 0,
    phrase: "WELCOME TO MY WORLD",
    phrasePos: 0,
    phraseWrong: new Set(),
    celebration: 0,
  };
  // Drill walks the neural engine's static letters (A-Y minus motion J/Z),
  // so N, P and Q are included — they were undetectable under pure rules.
  const DRILL_SEQ = [..."ABCDEFGHIKLMNOPQRSTUVWXY"];

  let stableLetter = null, stableSince = 0, holdTyped = false;
  let lastTypedAt = 0, lastMotionType = 0;
  const voter = new VOTE.SlidingVote(10, 0.6); // smooths per-frame recognition
  const cooldown = {};                 // letter -> timestamp when re-typable
  let motionPath = [];                 // [{x, y, t}] index-tip trail
  let soundOn = true, audioCtx = null;
  let frameCount = 0, fps = 0, fpsTimer = performance.now();
  let noHandFrames = 0;
  let lastGlyph = null, lastCls = null, flashAt = 0;

  // ---- learning library state (db.js) ----
  let library = new LEARN.Library();
  let learnedModel = library.toModel();
  let learnDirty = false;
  let saveTimer = null;
  let seedTarget = "A";
  let seedMsgTimer = null;
  let clearArmed = false, clearTimer = null;

  const REL_BADGE = { solid: "reliable", approx: "approx", motion: "trace it", no: "not detected" };

  /* Hand landmark drawing (self-contained — the old MediaPipe drawing_utils
     helper is gone now that we use the modern HandLandmarker task). */
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],              // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],              // index
    [5, 9], [9, 10], [10, 11], [11, 12],         // middle
    [9, 13], [13, 14], [14, 15], [15, 16],       // ring
    [13, 17], [17, 18], [18, 19], [19, 20],      // pinky
    [0, 17],                                     // palm heel
  ];
  function drawHand(ctx, lm) {
    const px = (i) => lm[i].x * ctx.canvas.width;
    const py = (i) => lm[i].y * ctx.canvas.height;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#22d3ee";
    ctx.lineJoin = "round";
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }
    ctx.fillStyle = "#f472b6";
    for (let i = 0; i < lm.length; i++) {
      ctx.beginPath();
      ctx.arc(px(i), py(i), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ---------------- build static UI ----------------
  const KEY_ROWS = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["I", "J", "K", "L", "M", "N", "O", "P"],
    ["Q", "R", "S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPACE"],
  ];
  KEY_ROWS.forEach((row) => {
    const div = document.createElement("div");
    div.className = "kb-row";
    row.forEach((ch) => {
      const b = document.createElement("button");
      b.className = "key" + (ch === "SPACE" ? " key-space" : "");
      b.textContent = ch === "SPACE" ? "␣ space" : ch;
      b.dataset.letter = ch;
      b.addEventListener("click", () => typeLetter(ch === "SPACE" ? " " : ch, true));
      div.appendChild(b);
    });
    keyboardEl.appendChild(div);
  });

  for (const ch of ALPHABET) {
    const info = ASL.LETTERS[ch];
    const cell = document.createElement("div");
    cell.className = "guide-cell rel-" + info.rel;
    cell.innerHTML =
      `<div class="g-emoji">${info.emoji || "·"}</div>` +
      `<div class="g-letter">${ch}</div>` +
      `<div class="g-desc">${info.desc}</div>` +
      `<div class="g-badge">${REL_BADGE[info.rel]}</div>`;
    guideGridEl.appendChild(cell);
  }

  // ---------------- status / helpers ----------------
  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", !!isError);
  }

  function resetStability() {
    stableLetter = null;
    stableSince = 0;
    holdTyped = false;
  }

  function tickSound(freq) {
    if (!soundOn) return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const t = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.1, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(t);
      osc.stop(t + 0.1);
    } catch (e) { /* audio not allowed yet */ }
  }

  // ---------------- learning library ----------------
  /* Persist the library (debounced) to IndexedDB — the in-browser DB. */
  function scheduleSave() {
    if (saveTimer) return;
    saveTimer = setTimeout(async () => {
      saveTimer = null;
      try { await LEARN.saveToIDB(library); }
      catch (e) { console.warn("Could not save learning library:", e); }
    }, 800);
  }

  /* Add one sample to the library and refresh the learned model. */
  function collectSample(letter, vec) {
    if (!letter || letter === " " || !vec) return;
    const res = library.add(letter, vec, Date.now());
    if (res.added) {
      learnedModel = library.toModel();
      learnDirty = true;
      scheduleSave();
      renderSeed();
    }
  }

  /* Load order: legacy My-signs model (migrate once) -> this browser's
     DB -> the shipped seed DB (single-file build inlines it, otherwise
     fetched next to the app). Everything merges; dedupe keeps it clean. */
  async function initLearn() {
    try {
      const raw = localStorage.getItem(LEGACY_MODEL_KEY);
      if (raw) {
        const legacy = KNN.Model.fromJSON(JSON.parse(raw));
        let n = 0;
        for (const s of legacy.samples) {
          if (library.add(s.letter, s.v, Date.now()).added) n++;
        }
        localStorage.removeItem(LEGACY_MODEL_KEY);
        if (n) console.log("Migrated", n, "legacy My-signs samples into the learning library");
      }
    } catch (e) { console.warn("Could not migrate legacy model:", e); }

    try { await LEARN.loadFromIDB(library); }
    catch (e) { console.warn("Could not load learning library:", e); }

    try {
      if (window.SEED_DB) {
        // Single-file build inlines the db directly (see tools/build-single.mjs).
        loadSeed(window.SEED_DB, "bundled db");
      } else {
        // Try each candidate independently so a missing/corrupt db.json
        // (or a fetch that throws, e.g. under file://) still falls back
        // to the shipped seed-db.json.
        for (const url of SEED_CANDIDATES) {
          try {
            const res = await fetch(url);
            if (res.ok) { loadSeed(await res.json(), url); break; }
          } catch (e) { console.warn("Seed DB " + url + " unavailable:", e); }
        }
      }
    } catch (e) { /* no db shipped — fine */ }

    learnedModel = library.toModel();
    learnDirty = true;
    renderSeed();
    scheduleSave();
  }

  function loadSeed(json, label) {
    if (!json || !json.samples || !json.samples.length) return;
    try {
      const seed = LEARN.Library.fromJSON(json);
      const added = LEARN.mergeInto(library, seed);
      if (added) console.log("Seed DB", label, "added", added, "samples");
    } catch (e) { console.warn("Bad seed DB (" + label + "):", e); }
  }

  // ---------------- seed trainer UI ----------------
  function renderSeed() {
    if (!seedSummary) return;
    const lc = library.counts();
    const letters = Object.keys(lc).length;
    const got = lc[seedTarget] || 0;
    seedSummary.textContent = `${library.total} samples · ${letters}/26 letters`;

    seedBig.textContent = seedTarget;
    seedName.innerHTML =
      `<b>${seedTarget}</b> · ${ASL.LETTERS[seedTarget].desc} — ` +
      (got >= SEED_GOAL
        ? `${got} samples, good. Keep adding variety or move on.`
        : `sign it and hold steady; each hold records (${got} so far).`);

    let dots = "";
    const n = Math.min(got, MAX_DOTS);
    for (let i = 0; i < n; i++) dots += `<span class="seed-dot fill"></span>`;
    if (got > MAX_DOTS) dots += `<span class="tc-extra">+${got - MAX_DOTS}</span>`;
    seedDots.innerHTML = dots;
  }

  function flashSeed(msg, ok = true) {
    seedSummary.textContent = msg;
    seedSummary.style.color = ok ? "var(--good)" : "var(--bad)";
    if (seedMsgTimer) clearTimeout(seedMsgTimer);
    seedMsgTimer = setTimeout(() => {
      seedSummary.style.color = "";
      renderSeed();
    }, 2600);
  }

  function stepSeed(dir) {
    const idx = ALPHABET.indexOf(seedTarget);
    seedTarget = ALPHABET[(idx + dir + 26) % 26];
    renderSeed();
  }

  seedPrev.addEventListener("click", () => stepSeed(-1));
  seedNext.addEventListener("click", () => stepSeed(1));

  /* The simple export mechanic: download the current database as db.json.
     Save it at the project root (or rebuild signtype.html) and every
     visitor starts pre-trained with your hand model. */
  seedExport.addEventListener("click", () => {
    if (!library.total) {
      flashSeed("Nothing to export yet — sign a letter above first", false);
      return;
    }
    const json = LEARN.exportJSON(library);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "db.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    flashSeed(`Exported ${library.total} samples → save as db.json at the project root to ship it`);
  });

  seedImport.addEventListener("click", () => seedImportFile.click());
  seedImportFile.addEventListener("change", async () => {
    const file = seedImportFile.files && seedImportFile.files[0];
    seedImportFile.value = "";
    if (!file) return;
    try {
      const seed = LEARN.importJSON(await file.text());
      const added = LEARN.mergeInto(library, seed);
      learnedModel = library.toModel();
      learnDirty = true;
      scheduleSave();
      renderSeed();
      flashSeed(`Imported ${seed.total} samples (${added} new to this browser)`);
    } catch (e) {
      flashSeed("Import failed — not a SignType learning DB", false);
    }
  });

  seedClear.addEventListener("click", () => {
    if (!clearArmed) {
      clearArmed = true;
      seedClear.classList.add("armed");
      seedClear.textContent = "Really clear?";
      clearTimer = setTimeout(disarmClear, 2600);
      return;
    }
    disarmClear();
    library.clear();
    learnedModel = library.toModel();
    learnDirty = true;
    scheduleSave();
    renderSeed();
    flashSeed("Learning library cleared in this browser");
  });
  function disarmClear() {
    clearArmed = false;
    seedClear.classList.remove("armed");
    seedClear.textContent = "🗑 Clear";
    if (clearTimer) { clearTimeout(clearTimer); clearTimer = null; }
  }

  // ---------------- typing ----------------
  function typeLetter(letter, manual) {
    const now = performance.now();
    lastTypedAt = now;
    if (!manual) cooldown[letter] = now;

    // feedback
    flashAt = now;
    currentLetterEl.classList.remove("flash");
    void currentLetterEl.offsetWidth; // restart animation
    currentLetterEl.classList.add("flash");
    tickSound(letter === " " ? 240 : 660);

    // apply to the active mode
    if (state.mode === "free") {
      state.typed += letter;
      updateOutputUI();
    } else if (state.mode === "alphabet") {
      if (letter === DRILL_SEQ[state.drillIdx]) {
        state.drillIdx++;
        if (state.drillIdx >= DRILL_SEQ.length) {
          state.drillIdx = 0;
          state.celebration = performance.now();
          tickSound(880);
          setTimeout(() => tickSound(1175), 120);
        }
      }
      updateOutputUI();
    } else if (state.mode === "phrase") {
      const expect = state.phrase[state.phrasePos];
      if (letter === expect) {
        state.phraseWrong.delete(state.phrasePos);
        state.phrasePos++;
        if (state.phrasePos >= state.phrase.length) {
          state.phrasePos = 0;
          state.phraseWrong.clear();
          state.celebration = performance.now();
          tickSound(880);
          setTimeout(() => tickSound(1175), 120);
        }
      } else {
        state.phraseWrong.add(state.phrasePos);
        tickSound(140);
      }
      updateOutputUI();
    }
  }

  function updateOutputUI() {
    // keyboard drill states
    document.querySelectorAll(".key").forEach((k) => {
      const ch = k.dataset.letter;
      const inDrill = state.mode === "alphabet" && DRILL_SEQ.indexOf(ch) !== -1;
      k.classList.toggle("target", inDrill && DRILL_SEQ[state.drillIdx] === ch);
      k.classList.toggle("done", inDrill && DRILL_SEQ.indexOf(ch) < state.drillIdx);
    });

    if (state.mode === "free") {
      typedBox.textContent = state.typed + "▍";
      typedSub.textContent = state.typed.length + " chars";
    } else if (state.mode === "alphabet") {
      typedBox.innerHTML = DRILL_SEQ.map((ch, i) =>
        `<span class="${i < state.drillIdx ? "done" : i === state.drillIdx ? "target" : "muted"}">${ch}</span>`
      ).join(" ");
      drillBar.style.width = (state.drillIdx / DRILL_SEQ.length * 100) + "%";
      const celeb = performance.now() - state.celebration < 1500;
      drillTarget.innerHTML = celeb
        ? "🎉 Alphabet done! Starting over…"
        : `Sign <b class="target-letter">${DRILL_SEQ[state.drillIdx]}</b>` +
          `<span class="muted"> · ${state.drillIdx} / ${DRILL_SEQ.length}</span>`;
      typedSub.textContent = state.drillIdx + " / " + DRILL_SEQ.length;
    } else { // phrase
      typedBox.innerHTML = state.phrase.split("").map((c, i) => {
        let cls = "ph";
        if (i < state.phrasePos) cls += " done";
        else if (i === state.phrasePos) cls += " current";
        if (state.phraseWrong.has(i)) cls += " wrong";
        return `<span class="${cls}">${c === " " ? "␣" : c}</span>`;
      }).join("");
      drillBar.style.width = (state.phrasePos / state.phrase.length * 100) + "%";
      const celeb = performance.now() - state.celebration < 1500;
      drillTarget.innerHTML = celeb
        ? "🎉 Phrase done! Starting over…"
        : `Type the phrase <span class="muted">· ${state.phrasePos} / ${state.phrase.length} chars</span>`;
      typedSub.textContent = state.phrasePos + " / " + state.phrase.length;
    }
  }

  function resetDrill() {
    state.drillIdx = 0;
    state.phrasePos = 0;
    state.phraseWrong = new Set();
    state.celebration = 0;
    updateOutputUI();
  }

  // ---------------- mode switching ----------------
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) =>
        b.classList.toggle("active", b === btn));
      state.mode = btn.dataset.mode;
      drillEl.classList.toggle("hidden", state.mode === "free");
      phraseRow.classList.toggle("hidden", state.mode !== "phrase");
      keyboardEl.classList.remove("hidden");
      if (state.mode === "phrase") phraseInput.value = state.phrase;
      resetDrill();
      resetStability();
      voter.clear();
    });
  });

  phraseStart.addEventListener("click", () => {
    const p = (phraseInput.value || "").toUpperCase().trim();
    if (p) {
      state.phrase = p;
      resetDrill();
    }
  });
  phraseInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") phraseStart.click();
  });

  copyBtn.addEventListener("click", async () => {
    const text = state.mode === "free" ? state.typed
      : state.mode === "phrase" ? state.phrase
      : DRILL_SEQ.join("");
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
    } catch (e) {
      copyBtn.textContent = "Failed";
    }
  });

  clearBtn.addEventListener("click", () => {
    if (state.mode === "free") {
      state.typed = "";
      updateOutputUI();
    } else {
      resetDrill();
    }
  });

  soundToggle.addEventListener("click", () => {
    soundOn = !soundOn;
    soundToggle.textContent = soundOn ? "🔊" : "🔇";
  });

  // ---------------- camera + tracking ----------------
  let landmarker = null;
  let lastVideoTime = -1;

  /* Modern MediaPipe Hand Landmarker task (GPU-accelerated, VIDEO mode).
     This replaces the old deprecated @mediapipe/hands "Hands" solution,
     which ran in IMAGE mode on the CPU and lost tracking easily. The task
     API tracks a hand between frames with a lightweight matcher and only
     re-runs palm detection when tracking fails, so detection is much more
     stable under motion, tilt and partial occlusion. The 21 normalized
     landmarks it emits use the same conventions as the old engine, so the
     whole recognition stack (rules + learned k-NN) works unchanged. */
  async function initCamera() {
    // 1) Open the webcam. Raw (non-mirrored) feed — we mirror on the canvas.
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    } catch (err) {
      console.error(err);
      setStatus("Camera unavailable", true);
      cameraError.classList.remove("hidden");
      return;
    }

    // 2) Load the Hand Landmarker task + its WASM runtime.
    try {
      setStatus("Loading hand-tracking model…");
      const vision = await import(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs"
      );
      const fileset = await vision.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      const makeOpts = (delegate) => ({
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate,
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      try {
        landmarker = await vision.HandLandmarker.createFromOptions(fileset, makeOpts("GPU"));
      } catch (e) {
        // No WebGL2 / GPU unsupported — fall back to CPU.
        landmarker = await vision.HandLandmarker.createFromOptions(fileset, makeOpts("CPU"));
      }
      setStatus("Ready — hold a sign to type");
      requestAnimationFrame(renderLoop);
    } catch (err) {
      console.error(err);
      setStatus("Hand-tracking engine failed to load (check internet)", true);
      cameraError.classList.remove("hidden");
    }
  }

  /* Drive the landmarker in VIDEO mode: one detectForVideo call per new
     video frame, with a monotonic timestamp (performance.now()). */
  function renderLoop() {
    if (!landmarker) return;
    const t = video.currentTime;
    if (t !== lastVideoTime) {
      lastVideoTime = t;
      try {
        onResults(landmarker.detectForVideo(video, performance.now()));
      } catch (e) { /* duplicate/edge frame — just skip it */ }
    }
    requestAnimationFrame(renderLoop);
  }

  /* The two-engine decision. Returns:
       { letter, via, conf, vec, hint, learnedOk, learnedLetter }
     letter: chosen letter or null; via: "agreed" | "rules" | "learned";
     vec: canonical vector for auto-collection; hint: unsure-pose text.
     Ladder (validated on real data):
       1. rules and learned agree  -> strongest signal
       2. learned >= LEARNED_OVERRIDE (>=94% precise) -> trust the crowd
       3. rules                    -> the tested default
       4. learned (gated 0.55/1.2) -> covers N/P/Q and user style
     The learned engine is a k-NN over a real-hand seed (shipped) plus
     everything collected locally, so it gets better with use. */
  function classifyFrame(flat, feats) {
    const rulesLetter = ASL.classify(feats);
    const vec = KNN.canonicalize(flat);
    let learned = null;
    if (vec) learned = learnedModel.classify(vec);
    const learnedOk = !!(learned && learned.letter &&
      learned.conf >= CONF_MIN && learned.nn <= NN_MAX);
    const learnedLetter = learned && learned.letter;

    if (rulesLetter && learnedOk && learnedLetter === rulesLetter) {
      return { letter: rulesLetter, via: "agreed", conf: learned.conf, vec,
        learnedOk, learnedLetter, hint: ASL.LETTERS[rulesLetter].desc };
    }
    if (learnedOk && learned.conf >= LEARNED_OVERRIDE) {
      return { letter: learnedLetter, via: "learned", conf: learned.conf, vec,
        learnedOk, learnedLetter,
        hint: `your ${learnedLetter} · learned from ${library.counts()[learnedLetter] || 0} samples — hold to type` };
    }
    if (rulesLetter) {
      return { letter: rulesLetter, via: "rules", conf: 0, vec,
        learnedOk, learnedLetter, hint: ASL.LETTERS[rulesLetter].desc };
    }
    if (learnedOk) {
      return { letter: learnedLetter, via: "learned", conf: learned.conf, vec,
        learnedOk, learnedLetter,
        hint: `your ${learnedLetter} · learned from ${library.counts()[learnedLetter] || 0} samples — hold to type` };
    }
    const hint = learned && learned.letter
      ? `Learned ${learned.letter} only ${Math.round(learned.conf * 100)}% — hold still or check the guide`
      : "Not recognized — check the guide";
    return { letter: null, via: "none", conf: 0, vec, learnedOk, learnedLetter, hint };
  }

  function onResults(result) {
    frameCount++;
    const now = performance.now();

    const lm = result && result.landmarks && result.landmarks[0];

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Mirror the live frame (selfie view, same as the old selfieMode). The
    // camera feed is raw/un-mirrored; the canvas shows the mirrored copy.
    ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    if (lm) {
      // Mirror landmark x too, so the skeleton lines up with the frame.
      drawHand(ctx, lm.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z })));
      noHandFrames = 0;
      handHint.classList.remove("show");
      processFrame(lm, now);
    } else {
      noHandFrames++;
      voter.push(null); // decay the vote while the hand is gone
      if (noHandFrames > 12) handHint.classList.add("show");
      resetStability();
      setCurrentDisplay("—", "", "Hold a sign steady to type it");
      featuresEl.innerHTML = "";
      document.querySelectorAll(".key").forEach((k) => k.classList.remove("active"));
    }
    ctx.restore();

    if (now - fpsTimer > 1000) {
      fps = Math.round((frameCount * 1000) / (now - fpsTimer));
      frameCount = 0;
      fpsTimer = now;
    }
  }

  function setCurrentDisplay(glyph, cls, name) {
    const now = performance.now();
    if (now - flashAt < 450) cls = cls ? "flash " + cls : "flash";
    if (glyph !== lastGlyph || cls !== lastCls) {
      currentLetterEl.textContent = glyph;
      currentLetterEl.className = "current-letter" + (cls ? " " + cls : "");
      lastGlyph = glyph;
      lastCls = cls;
    }
    currentNameEl.textContent = name;
  }

  function processFrame(lm, now) {
    const flat = lm.map((p) => [p.x, p.y, p.z]);
    const feats = ASL.analyze(flat);
    const read = classifyFrame(flat, feats);

    let typedLetter = null;
    let glyph = "?", cls = "unknown", name = read.hint;

    // index-tip trail for J / Z (motion letters are rule-based)
    if (feats.ext.index && !feats.ext.middle && !feats.ext.ring && !feats.ext.pinky) {
      motionPath.push({ x: lm[ASL.L.INDEX_TIP].x, y: lm[ASL.L.INDEX_TIP].y, t: now });
    }
    while (motionPath.length && now - motionPath[0].t > 1500) motionPath.shift();

    // ---- what is the hand signing right now? ----
    const openHand = feats.pattern === "1111" && feats.thumbOut;
    if (openHand) {
      typedLetter = " "; // " " => space
      glyph = "5"; cls = "space";
      name = "open hand · types space";
    } else if (read.letter) {
      typedLetter = read.letter;
      glyph = read.letter;
      cls = read.via === "learned" ? "learned" : "";
      name = read.hint;
    }

    // ---- motion letters (J / Z) ----
    evaluateMotion(feats, now);

    // ---- temporal smoothing ----
    // Per-frame recognition flickers between look-alike letters (A/S/E,
    // H/U, V/K...). A single misread frame used to reset the whole hold
    // timer — now a letter must win a majority of the last 10 frames, so
    // glitches can't break a hold or type the wrong letter.
    const rawLetter = typedLetter;
    const voted = voter.push(rawLetter);
    if (voted !== null) {
      typedLetter = voted;
      if (voted !== rawLetter) {
        // mid-transition: the vote still points at the previous letter
        glyph = voted === " " ? "5" : voted;
        cls = voted === " " ? "space" : "";
        name = voted === " " ? "open hand · types space" : ASL.LETTERS[voted].desc;
      }
    } else if (rawLetter !== null) {
      // hand visible but no majority yet (switching letters)
      glyph = "…"; cls = "unknown";
      name = "Reading your hand…";
      typedLetter = null;
    }

    setCurrentDisplay(glyph, cls, name);

    document.querySelectorAll(".key").forEach((k) => {
      k.classList.toggle("active",
        k.dataset.letter === (typedLetter === " " ? "SPACE" : typedLetter));
    });

    // ---- stability → type ----
    if (typedLetter !== null) {
      if (typedLetter === stableLetter && !holdTyped) {
        if (stableSince === 0) stableSince = now;
        const canType = now - lastMotionType > 800 &&
          now - (cooldown[typedLetter] || 0) > 900 &&
          now - lastTypedAt > 250;
        if (now - stableSince >= 500 && canType) {
          typeLetter(typedLetter, false);
          cooldown[typedLetter] = now;
          holdTyped = true; // one letter per hold
        }
      } else if (typedLetter !== stableLetter) {
        // hold just started: the vote majority is solid, so this is a
        // good moment to learn from the sign. Collect when the reading is
        // strong — learned at >=75% (98% precise) or rules backed by the
        // learned DB / no learned opinion yet (bootstrap).
        if (rawLetter !== null && rawLetter === typedLetter && typedLetter !== " ") {
          const strongLearned = read.via === "learned" && read.conf >= LEARNED_OVERRIDE;
          const rulesTrusted = read.via === "rules" &&
            (!read.learnedOk || read.learnedLetter === typedLetter);
          if (strongLearned || rulesTrusted ||
              (seedTrain.open && typedLetter === seedTarget)) {
            collectSample(typedLetter, read.vec);
          }
        }
        stableLetter = typedLetter;
        stableSince = now;
        holdTyped = false;
      }
    } else {
      resetStability();
    }

    // ---- live feature readout ----
    renderFeatures(feats);
  }

  function evaluateMotion(feats, now) {
    const pointing = feats.pattern === "1000" && !feats.thumbOut;
    if (!pointing) return;
    const path = motionPath.filter((p) => now - p.t <= 900);
    if (path.length < 14) return;
    const letter = ASL.classifyMotion(path);
    if (letter && now - (cooldown[letter] || 0) > 1500) {
      cooldown[letter] = now;
      lastMotionType = now;
      typeLetter(letter, false);
      motionPath.length = 0;
    }
  }

  function renderFeatures(feats) {
    const names = ["index", "middle", "ring", "pinky"];
    const dots = names.map((n) =>
      `<span class="f-${feats.ext[n] ? "on" : "off"}" title="${n}">${feats.ext[n] ? "●" : "○"}</span>`).join("");
    const thumb = `<span class="f-${feats.thumbOut ? "on" : "off"}" title="thumb">${feats.thumbOut ? "●" : "○"}</span>`;
    const learned = library.total
      ? `<span class="f-label">· learned</span> ${library.total}/${Object.keys(library.counts()).length}`
      : "";
    featuresEl.innerHTML =
      `<span class="f-label">pattern ${feats.pattern}</span> ${dots}${thumb}` +
      `<span class="f-label">· thumb ${feats.thumbOut ? "out" : "folded"}</span>` +
      learned +
      `<span class="f-fps">🧠 learned · ${fps} fps</span>`;
  }

  // ---------------- go ----------------
  renderSeed();
  updateOutputUI();
  initLearn();
  initCamera();
})();