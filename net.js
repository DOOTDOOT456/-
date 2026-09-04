/* Copyright © 2026 [Your Full Name]. All rights reserved.

   Neural engine: a small dense network that maps 21 MediaPipe hand
   landmarks (x, y, z each, 63 inputs, flattened as-is) to an ASL static
   letter (A-Y, no J/Z — those are traced). Pure JS, no dependencies.

   The weights in asl-net.json come from the ASL Alphabet Classifier by
   AmimulBmeIU (MIT license, validation accuracy 94.15%):
     https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier
   They were exported from its TensorFlow.js build by tools/export-net.mjs,
   which byte-slices the weight shard; tools/parity-tfjs.mjs confirms the
   sliced weights reproduce the original model's outputs (max diff < 1e-6).

   Architecture: Dense 128 relu -> Dropout(0.3, inference no-op) ->
   Dense 64 relu -> Dropout(0.2, no-op) -> Dense 24 softmax.
   Keras channel-last conventions: Dense kernel [in, out]. */
"use strict";

const NET = (function () {

  function dense(vec, layer) {
    const { w, b } = layer;
    const outN = layer.units || layer.out;
    const out = new Float32Array(outN);
    for (let o = 0; o < outN; o++) {
      let acc = b[o];
      for (let i = 0; i < vec.length; i++) acc += vec[i] * w[i * outN + o];
      out[o] = acc;
    }
    return out;
  }

  function reluInPlace(a) {
    for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
    return a;
  }

  function softmax(logits) {
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let sum = 0;
    const p = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      p[i] = Math.exp(logits[i] - mx);
      sum += p[i];
    }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  /* Predict from a flat 63-value landmark vector ([x,y,z]*21, mediapipe
     normalized coordinates, z wrist-relative). Returns
     { letter, index, conf, probs } or null for a bad input length. */
  function predict(model, x63) {
    if (!model || !x63 || x63.length !== 63) return null;
    let h = new Float32Array(63);
    h.set(x63);
    for (const layer of model.layers) {
      if (layer.type !== "dense") continue; // dropout is a training no-op
      h = dense(h, layer);
      if (layer.activation === "relu") reluInPlace(h);
      else if (layer.activation === "softmax") {
        const probs = softmax(h);
        let best = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
        return {
          letter: model.letters[best],
          index: best,
          conf: probs[best],
          probs,
        };
      }
    }
    return null;
  }

  function load(json) {
    if (!json || !json.layers || !json.letters) {
      throw new Error("net: model JSON missing layers or letters");
    }
    return json;
  }

  return { load, predict, softmax };
})();
