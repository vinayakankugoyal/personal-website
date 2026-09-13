// worker.js — owns the model. Requests are serialized: one asr() at a time.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4";

let asr = null;
let queue = Promise.resolve();

self.onmessage = ({ data }) => {
    if (data.type === "load")       queue = queue.then(load);
    if (data.type === "transcribe") queue = queue.then(() => transcribe(data));
};

async function load() {
    if (asr) return self.postMessage({ type: "ready" });
    asr = await pipeline("automatic-speech-recognition", "onnx-community/whisper-base.en_timestamped", {
        device: "webgpu",
        dtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
        progress_callback: (p) => self.postMessage({ type: "progress", ...p }),
    });
    self.postMessage({ type: "ready" });
}

async function transcribe(data) {
    try {
        // chunking handles recordings longer than 30 s
        const result = await asr(data.audio, { chunk_length_s: 30, stride_length_s: 5 });
        self.postMessage({ type: "result", result });
    } catch (e) {
        self.postMessage({ type: "error", message: e.message });
    }
}