// worker.js — owns the model. Requests are serialized: one asr() at a time.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4";

let asr = null;
let queue = Promise.resolve();

self.onmessage = ({ data }) => {
    if (data.type === "load")       queue = queue.then(load);
    if (data.type === "transcribe") queue = queue.then(() => transcribe(data));
};

// Returns a reason WebGPU can't be used, or null if there is a real GPU behind it.
async function webgpuProblem() {
    if (!navigator.gpu) return "This browser does not support WebGPU.";
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (!adapter) return "WebGPU is available but no GPU adapter was found.";
    // isFallbackAdapter moved from the adapter to adapter.info; check both.
    const info = adapter.info ?? {};
    const software =
        adapter.isFallbackAdapter ||
        info.isFallbackAdapter ||
        /swiftshader|llvmpipe|software/i.test(`${info.architecture} ${info.device} ${info.description}`);
    if (software) return "WebGPU is running in software here (no hardware acceleration), which is too slow for Whisper. Check that hardware acceleration is enabled in your browser settings.";
    return null;
}

async function load() {
    if (asr) return self.postMessage({ type: "ready" });
    try {
        const problem = await webgpuProblem();
        if (problem) return self.postMessage({ type: "error", message: problem });
        asr = await pipeline("automatic-speech-recognition", "onnx-community/whisper-base.en_timestamped", {
            device: "webgpu",
            dtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
            progress_callback: (p) => self.postMessage({ type: "progress", ...p }),
        });
        self.postMessage({ type: "ready" });
    } catch (e) {
        self.postMessage({ type: "error", message: `Could not load the speech model: ${e.message}` });
    }
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
