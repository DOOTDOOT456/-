/* Copyright © 2026 [Your Full Name]. All rights reserved.

   SignType — app glue: webcam, hand tracking, typing logic, UI.
   Recognition engines:
     - neural (net.js + asl-net.json): a CNN trained on the public
       Sign Language MNIST dataset — the default engine for static
       letters in Free/Alphabet/Phrase modes. Covers A-Y incl. N/P/Q.
     - rules (asl.js): kept as a fallback when the net is unsure, and
       for open-hand space and J/Z motion tracing.
     - k-NN (knn.js): user-trained model in the "My signs" mode. */
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
  const trainPanel = $("train-panel"), tabRecord = $("tab-record"), tabPractice = $("tab-practice");
  const recordView = $("record-view"), practiceView = $("practice-view");
  const trainBig = $("train-big"), trainTargetName = $("train-target-name"), trainDots = $("train-dots");
  const trainGrid = $("train-grid"), captureBtn = $("capture-btn"), skipBtn = $("skip-btn");
  const undoBtn = $("undo-btn"), clearModelBtn = $("clear-model-btn");
  const practiceLive = $("practice-live"), practiceReady = $("practice-ready"), trainFoot = $("train-foot");

  // ---------------- constants ----------------
  const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const MODEL_KEY = "signtype-knn-model-v1";
  const NET_CONF_MIN = 0.6;    // neural engine: act on a class only above this
  const NET_ONLY = new Set(["N", "P", "Q"]); // letters the rules can't produce
  const REC_GOAL = 3;          // guided recording stops here and auto-advances
  const REC_MAX = 8;           // hard cap of samples per letter
  const CONF_MIN = 0.55;       // k-NN confidence gate (0..1)
  const NN_MAX = 1.2;          // k-NN nearest-neighbor distance gate (unknown poses)
  const STEADY_RMS = 0.06;     // canonical per-coordinate jitter below this = steady
  const STEADY_MS = 600;       // hold steady this long to auto-record a sample
  const CAP_GAP_MS = 900;      // minimum gap between recorded samples

  // ---------------- state ----------------
  const state = {
    mode: "free",
    typed: "",
    drillIdx: 0,
    phrase: "WELCOME TO MY WORLD",
    phrasePos: 0,
    phraseWrong: new Set(),
    celebration: 0,
    trainTab: "record",        // "record" | "practice"
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
  let lastFeatures = null;
  let lastGlyph = null, lastCls = null, flashAt = 0;

  // ---- neural engine state (net.js + asl-net.json) ----
  let netModel = null;
  let netStatus = "loading";   // "loading" | "ready" | "error"

  // ---- train-your-own model state ----
  let model = new KNN.Model({ k: 7, minSamples: 2 });
  let trainTarget = "A";
  let trainCells = {};                 // letter -> cell element
  let prevVec = null;                  // previous frame's canonical vector
  let lastVec = null;                  // latest canonical vector (manual capture)
  let recArmed = true, recSteadySince = 0, lastCapAt = 0;
  let footTimer = null;
  let clearArmed = false, clearTimer = null;

  const REL_BADGE = { solid: "reliable", approx: "approx", motion: "trace it", no: "not detected" };

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

  for (const ch of ALPHABET) {
    const cell = document.createElement("div");
    cell.className = "train-cell";
    cell.innerHTML = `<span class="tc-letter">${ch}</span><span class="tc-n"></span>`;
    cell.addEventListener("click", () => {
      if (state.trainTab === "practice") setTrainTab("record");
      trainTarget = ch;
      renderTrainUI();
      resetRecorder();
    });
    trainCells[ch] = cell;
    trainGrid.appendChild(cell);
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
    lastFeatures = null;
  }

  function resetRecorder() {
    recArmed = true;
    recSteadySince = 0;
    prevVec = null;
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

  // ---------------- model persistence ----------------
  function saveModel() {
    try { localStorage.setItem(MODEL_KEY, JSON.stringify(model.toJSON())); }
    catch (e) { console.warn("Could not save model:", e); }
  }

  function loadModel() {
    try {
      const raw = localStorage.getItem(MODEL_KEY);
      if (!raw) return;
      const loaded = KNN.Model.fromJSON(JSON.parse(raw));
      if (loaded.total > 0) {
        model = loaded;
        console.log("Loaded saved My-signs model:", model.total, "samples");
      }
    } catch (e) { console.warn("Could not load saved model:", e); }
  }

  // ---------------- train UI ----------------
  function flashFoot(msg, ok = true) {
    trainFoot.innerHTML = `<span class="${ok ? "ok" : "bad"}">${msg}</span>`;
    if (footTimer) clearTimeout(footTimer);
    footTimer = setTimeout(renderTrainUI, 2600);
  }

  function renderTrainUI() {
    const counts = model.counts();
    const got = counts[trainTarget] || 0;

    // target header
    trainBig.textContent = trainTarget;
    const desc = ASL.LETTERS[trainTarget].desc;
    if (got >= REC_GOAL) {
      trainTargetName.innerHTML =
        `<b>${trainTarget}</b> has ${got} samples — enough. Click a letter below to add more, ` +
        `or press <b>Skip ➜</b> for the next one that still needs samples.`;
    } else {
      const n = got + 1;
      const p = n === 1 ? "st" : n === 2 ? "nd" : "rd";
      trainTargetName.innerHTML =
        `Sign <b>${trainTarget}</b> — hold it steady and it records itself ` +
        `(sample ${n}${p} of ${REC_GOAL}). ${desc}`;
    }
    let dots = "";
    for (let i = 0; i < REC_GOAL; i++) {
      dots += `<span class="train-dot${i < Math.min(got, REC_GOAL) ? " fill" : ""}"></span>`;
    }
    if (got > REC_GOAL) dots += `<span class="tc-extra">+${got - REC_GOAL}</span>`;
    trainDots.innerHTML = dots;

    // letter grid
    for (const ch of ALPHABET) {
      const cell = trainCells[ch];
      const c = counts[ch] || 0;
      cell.classList.toggle("sel", ch === trainTarget);
      cell.classList.toggle("have", c > 0);
      cell.classList.toggle("done", c >= REC_GOAL);
      cell.querySelector(".tc-n").textContent = c > 0 ? c : "";
    }

    // footer + practice-ready line
    const ready = model.activeLetters();
    practiceReady.textContent = "ready: " + (ready.join(" ") || "none yet — record in the Record tab");
    const allDone = ALPHABET.split("").every((ch) => (counts[ch] || 0) >= REC_GOAL);
    if (allDone) {
      trainFoot.innerHTML = `<span class="ok">🎉 All 26 letters have ${REC_GOAL}+ samples. Switch to Practice and sign!</span>`;
    } else if (model.total === 0) {
      trainFoot.innerHTML = `<span class="bad">Your model is empty — sign the <b>${trainTarget}</b> shown above and hold still.</span>`;
    } else {
      trainFoot.innerHTML =
        `<span class="ok">💾 auto-saved in this browser</span>` +
        `<span>${model.total} samples · ${ready.length}/26 letters ready</span>` +
        `<span class="muted">undo or clear to fix mistakes</span>`;
    }
    if (footTimer) { clearTimeout(footTimer); footTimer = null; } // a fresh render wins over flash
  }

  function setTrainTab(tab) {
    state.trainTab = tab;
    tabRecord.classList.toggle("active", tab === "record");
    tabPractice.classList.toggle("active", tab === "practice");
    recordView.classList.toggle("hidden", tab !== "record");
    practiceView.classList.toggle("hidden", tab !== "practice");
    resetRecorder();
    if (tab === "practice") {
      const ready = model.activeLetters();
      practiceLive.className = "practice-live dim";
      practiceLive.textContent = ready.length
        ? "Hold a sign from your model and it types it below."
        : "No letters ready yet — record at least 2 samples per letter, then come back here.";
    }
  }

  function selectCaptureButton() {
    if (!lastVec) { flashFoot("Show your hand in the camera first", false); return; }
    if ((model.counts()[trainTarget] || 0) >= REC_MAX) {
      flashFoot(`Max ${REC_MAX} samples for ${trainTarget} — undo or clear some first`, false);
      return;
    }
    captureSample(lastVec, performance.now());
  }

  function advanceTarget(announce) {
    const counts = model.counts();
    const need = ALPHABET.split("").filter((ch) => (counts[ch] || 0) < REC_GOAL);
    if (!need.length) {
      flashFoot("🎉 All letters trained — switch to Practice!");
      renderTrainUI();
      return;
    }
    const idx = ALPHABET.indexOf(trainTarget);
    for (let i = 1; i <= 26; i++) {
      const ch = ALPHABET[(idx + i) % 26];
      if ((counts[ch] || 0) < REC_GOAL) { trainTarget = ch; break; }
    }
    renderTrainUI();
    if (announce) flashFoot(`Next up: ${trainTarget} — sign it and hold steady`);
  }

  // ---------------- capture logic ----------------
  function rmsDiff(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; sum += t * t; }
    return Math.sqrt(sum / a.length);
  }

  /* Feed one frame's canonical vector into the steady-hold detector.
     A steady hold of STEADY_MS auto-records one sample for the target;
     the hand must move again (or leave frame) before another can record. */
  function captureTick(vec, now) {
    if (!vec) return;
    if (prevVec) {
      const rms = rmsDiff(prevVec, vec);
      if (rms > STEADY_RMS) {
        recSteadySince = 0;
        recArmed = true;
      } else if (recArmed && now - lastCapAt >= CAP_GAP_MS &&
                 (model.counts()[trainTarget] || 0) < REC_GOAL) {
        if (recSteadySince === 0) {
          recSteadySince = now;
        } else if (now - recSteadySince >= STEADY_MS) {
          captureSample(vec, now);
        }
      }
    }
    prevVec = vec;
  }

  function captureSample(vec, now) {
    const letter = trainTarget;
    const before = model.counts()[letter] || 0;
    lastCapAt = now;
    recArmed = false;          // must move the hand before the next capture
    recSteadySince = 0;
    model.add(letter, vec);
    saveModel();
    tickSound(520 + Math.min(before, 3) * 130);
    if (before + 1 >= REC_GOAL) {
      tickSound(880);
      setTimeout(() => tickSound(1175), 110);
    }
    renderTrainUI();
    if (before + 1 >= REC_GOAL) advanceTarget(false);
    else flashFoot(`${letter} sample ${before + 1}/${REC_GOAL} recorded 💾`);
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
    if (state.mode === "free" || state.mode === "train") {
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

    if (state.mode === "free" || state.mode === "train") {
      typedBox.textContent = state.typed + "▍";
      typedSub.textContent = state.mode === "train"
        ? state.typed.length + " chars · your model"
        : state.typed.length + " chars";
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
      drillEl.classList.toggle("hidden", state.mode === "free" || state.mode === "train");
      phraseRow.classList.toggle("hidden", state.mode !== "phrase");
      keyboardEl.classList.toggle("hidden", state.mode === "train");
      trainPanel.classList.toggle("hidden", state.mode !== "train");
      if (state.mode === "phrase") phraseInput.value = state.phrase;
      if (state.mode === "train") {
        renderTrainUI();
        setTrainTab(state.trainTab);
      }
      resetDrill();
      resetStability();
      voter.clear();
    });
  });

  tabRecord.addEventListener("click", () => setTrainTab("record"));
  tabPractice.addEventListener("click", () => setTrainTab("practice"));
  captureBtn.addEventListener("click", selectCaptureButton);
  skipBtn.addEventListener("click", () => advanceTarget(true));

  undoBtn.addEventListener("click", () => {
    const removed = model.removeLast();
    if (!removed) { flashFoot("Nothing to undo yet", false); return; }
    trainTarget = removed.letter;
    saveModel();
    resetRecorder();
    renderTrainUI();
    flashFoot(`Removed one ${removed.letter} sample (${model.counts()[removed.letter] || 0} left)`);
  });

  clearModelBtn.addEventListener("click", () => {
    if (!clearArmed) {
      clearArmed = true;
      clearModelBtn.classList.add("armed");
      clearModelBtn.textContent = "Really clear?";
      clearTimer = setTimeout(disarmClear, 2600);
      return;
    }
    disarmClear();
    model.clear();
    saveModel();
    trainTarget = "A";
    resetRecorder();
    renderTrainUI();
    flashFoot("Model cleared — start recording fresh");
  });
  function disarmClear() {
    clearArmed = false;
    clearModelBtn.classList.remove("armed");
    clearModelBtn.textContent = "🗑 Clear";
    if (clearTimer) { clearTimeout(clearTimer); clearTimer = null; }
  }

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
    const text = (state.mode === "free" || state.mode === "train") ? state.typed
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
    if (state.mode === "free" || state.mode === "train") {
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
  async function initCamera() {
    if (typeof Hands === "undefined" || typeof Camera === "undefined") {
      setStatus("Hand-tracking libraries failed to load (check internet)", true);
      cameraError.classList.remove("hidden");
      return;
    }
    try {
      const hands = new Hands({
        locateFile: (f) => "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" + f,
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
        selfieMode: true,
      });
      hands.onResults(onResults);
      const camera = new Camera(video, {
        onFrame: async () => { await hands.send({ image: video }); },
        width: 640,
        height: 480,
      });
      setStatus("Starting camera…");
      await camera.start();
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      setStatus("Ready — hold a sign to type");
    } catch (err) {
      console.error(err);
      setStatus("Camera unavailable", true);
      cameraError.classList.remove("hidden");
    }
  }

  // ---------------- neural engine ----------------
  async function initNet() {
    try {
      const res = await fetch("asl-net.json");
      if (!res.ok) throw new Error("HTTP " + res.status);
      netModel = NET.load(await res.json());
      netStatus = "ready";
      console.log("Neural engine ready:", netModel.letters.length, "letters");
    } catch (e) {
      netStatus = "error";
      console.warn("Neural engine unavailable (", e.message, ") — falling back to rules only");
    }
  }

  /* Run the neural engine on this frame's landmarks: flatten the 21
     mediapipe points [x, y, z] in order (as the model was trained) and
     classify. Returns { letter, conf, probs } or null when no model. */
  function predictNet(lm) {
    if (!netModel) return null;
    const v = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      v[i * 3] = lm[i].x;
      v[i * 3 + 1] = lm[i].y;
      v[i * 3 + 2] = lm[i].z;
    }
    return NET.predict(netModel, v);
  }

  function onResults(results) {
    frameCount++;
    const now = performance.now();

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    const lm = results.multiHandLandmarks && results.multiHandLandmarks[0];
    if (lm) {
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: "#22d3ee", lineWidth: 3 });
      drawLandmarks(ctx, lm, { color: "#f472b6", lineWidth: 2, radius: 4 });
      noHandFrames = 0;
      handHint.classList.remove("show");
      processFrame(lm, now);
    } else {
      noHandFrames++;
      voter.push(null); // decay the vote while the hand is gone
      if (noHandFrames > 12) handHint.classList.add("show");
      resetStability();
      prevVec = null;
      lastVec = null;
      recSteadySince = 0;
      const noHandMsg = state.mode === "train"
        ? "Show a hand to record or practice"
        : "Hold a sign steady to type it";
      setCurrentDisplay("—", "", noHandMsg);
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

  // ---- train-mode frame: k-NN recognition + guided recording ----
  function processTrainFrame(flat, now) {
    const vec = KNN.canonicalize(flat);
    lastVec = vec;
    let res = null, letter = null, glyph = "—", cls = "", name = "Hand not tracked";

    if (vec) {
      res = model.classify(vec);
      const confident = res.letter && res.conf >= CONF_MIN && res.nn <= NN_MAX;
      letter = confident ? res.letter : null;
      if (letter) {
        glyph = letter;
        cls = "trained";
        name = `Your ${letter} · ${Math.round(res.conf * 100)}% confident — hold to type`;
      } else {
        glyph = "?";
        cls = "unknown";
        name = model.activeLetters().length
          ? `Not confident (${res.letter ? Math.round(res.conf * 100) + "%" : "no samples close"}) — hold still or record more`
          : "No trained letters yet — record samples first";
      }
    } else {
      lastVec = null;
    }

    // recording happens only in the Record tab and only for the target letter
    if (state.trainTab === "record" && vec) captureTick(vec, now);

    // live model opinion (visible under Practice)
    if (res && res.letter) {
      practiceLive.textContent = letter
        ? `Sees ${res.letter} · ${Math.round(res.conf * 100)}% — hold steady to type`
        : `${res.letter}? only ${Math.round(res.conf * 100)}% — hold still or add more samples`;
      practiceLive.className = "practice-live" + (letter ? "" : " warn");
    }

    return { typedLetter: letter, glyph, cls, name };
  }

  function processFrame(lm, now) {
    const flat = lm.map((p) => [p.x, p.y, p.z]);
    const feats = ASL.analyze(flat);
    lastFeatures = feats;
    const trainMode = state.mode === "train";

    let typedLetter = null;
    let glyph = "?", cls = "unknown", name = "Not recognized — check the guide";

    if (trainMode) {
      const out = processTrainFrame(flat, now);
      typedLetter = out.typedLetter;
      glyph = out.glyph;
      cls = out.cls;
      name = out.name;
    } else {
      // index-tip trail for J / Z
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
      } else {
        // Engines: the neural net (trained on the public ASL Alphabet
        // dataset, covers all static letters incl. N/P/Q) and the tested
        // rule classifier. The net only types when it is confident AND it
        // agrees with the rules, the rules are silent, or it claims a
        // letter the rules cannot produce (N/P/Q) — so the neural engine
        // can only add letters, never override a rules reading.
        const rulesLetter = ASL.classify(feats);
        const nres = predictNet(lm);
        let letter = null, via = "rules", confPct = 0;
        if (nres) {
          const netConfident = nres.conf >= NET_CONF_MIN;
          const netOnly = NET_ONLY.has(nres.letter);
          const agree = rulesLetter === nres.letter;
          if (netConfident && (netOnly || agree || !rulesLetter)) {
            letter = nres.letter;
            via = "net";
            confPct = nres.conf;
          } else if (rulesLetter) {
            letter = rulesLetter;
          } else {
            glyph = "?"; cls = "unknown";
            name = `Net: ${nres.letter} at ${Math.round(nres.conf * 100)}% — hold still or adjust your hand`;
          }
        } else {
          letter = rulesLetter; // net unavailable (offline): rules as before
        }
        if (letter) {
          typedLetter = letter;
          glyph = letter; cls = "";
          name = via === "net"
            ? `${letter} · ${Math.round(confPct * 100)}% confident — hold to type`
            : ASL.LETTERS[letter].desc;
        }
      }

      // ---- motion letters (J / Z) ----
      evaluateMotion(feats, now);
    }

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
        name = trainMode
          ? "holding your " + voted + " — keep still"
          : voted === " " ? "open hand · types space" : ASL.LETTERS[voted].desc;
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

    // ---- stability → type (disabled while recording samples) ----
    const allowTyping = !(trainMode && state.trainTab === "record");
    if (allowTyping && typedLetter !== null) {
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
    if (state.mode === "train") {
      const ready = model.activeLetters().join("");
      featuresEl.innerHTML =
        `<span class="f-label">model</span> ${model.total} samples` +
        `<span class="f-label">· ready</span> ${ready || "—"}` +
        `<span class="f-fps">${fps} fps</span>`;
      return;
    }
    const names = ["index", "middle", "ring", "pinky"];
    const dots = names.map((n) =>
      `<span class="f-${feats.ext[n] ? "on" : "off"}" title="${n}">${feats.ext[n] ? "●" : "○"}</span>`).join("");
    const thumb = `<span class="f-${feats.thumbOut ? "on" : "off"}" title="thumb">${feats.thumbOut ? "●" : "○"}</span>`;
    const engine = netStatus === "ready" ? "🧠 neural" : netStatus === "error" ? "rules" : "🧠 loading…";
    featuresEl.innerHTML =
      `<span class="f-label">pattern ${feats.pattern}</span> ${dots}${thumb}` +
      `<span class="f-label">· thumb ${feats.thumbOut ? "out" : "folded"}</span>` +
      `<span class="f-fps">${engine} · ${fps} fps</span>`;
  }

  // ---------------- go ----------------
  loadModel();
  renderTrainUI();
  updateOutputUI();
  initNet();
  initCamera();
})();
