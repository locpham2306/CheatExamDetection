"use strict";

importScripts("/vendor/onnxruntime/ort.all.min.js");

const MODEL_INPUT_SIZE = 640;
const PERSON_CONFIDENCE_THRESHOLD = 0.25;
const NMS_THRESHOLD = 0.45;
const KEYPOINT_COUNT = 17;
const RECORD_SIZE = 5 + KEYPOINT_COUNT * 3;

let session = null;
let inputName = null;
let outputName = null;
let activeProvider = "wasm";

self.onmessage = async (event) => {
    const message = event.data || {};

    try {
        if (message.action === "INIT") {
            await initializePose(message.modelUrl);
            return;
        }

        if (message.action === "INFERENCE") {
            await runInference(message.tensorData, message.dims, message.meta || null);
        }
    } catch (error) {
        self.postMessage({
            status: "ERROR",
            error: error && error.message ? error.message : String(error)
        });
    }
};

async function initializePose(modelUrl) {
    if (!modelUrl) throw new Error("Missing pose model URL");

    ort.env.wasm.wasmPaths = "/vendor/onnxruntime/";
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";
    ort.env.logLevel = "warning";

    try {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all"
        });
        activeProvider = "webgpu";
    } catch (webgpuError) {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all"
        });
        activeProvider = "wasm";
    }

    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    self.postMessage({ status: activeProvider === "webgpu" ? "READY_WEBGPU" : "READY_WASM" });
}

async function runInference(tensorData, dims, meta) {
    if (!session) throw new Error("Pose session is not initialized");

    const inputTensor = new ort.Tensor("float32", tensorData, dims);
    const outputs = await session.run({ [inputName]: inputTensor });
    const output = outputs[outputName] || outputs[Object.keys(outputs)[0]];
    const people = decodePoseOutput(output);
    const flat = [];

    for (const person of people) {
        flat.push(person.x1, person.y1, person.x2, person.y2, person.confidence);
        for (const point of person.keypoints) {
            flat.push(point.x, point.y, point.confidence);
        }
    }

    self.postMessage({
        status: "RESULT",
        provider: activeProvider,
        recordSize: RECORD_SIZE,
        meta,
        data: flat
    });
}

function decodePoseOutput(output) {
    const data = output.data;
    const dims = output.dims || [];
    if (dims.length !== 3) return [];

    const attributeFirst = dims[1] <= dims[2];
    const attributes = attributeFirst ? dims[1] : dims[2];
    const candidates = attributeFirst ? dims[2] : dims[1];
    if (attributes < RECORD_SIZE || candidates < 1) return [];

    const valueAt = (candidate, attribute) => attributeFirst
        ? Number(data[attribute * candidates + candidate])
        : Number(data[candidate * attributes + attribute]);
    const people = [];

    for (let candidate = 0; candidate < candidates; candidate += 1) {
        const confidence = valueAt(candidate, 4);
        if (!Number.isFinite(confidence) || confidence < PERSON_CONFIDENCE_THRESHOLD) continue;

        let cx = valueAt(candidate, 0);
        let cy = valueAt(candidate, 1);
        let width = valueAt(candidate, 2);
        let height = valueAt(candidate, 3);

        if ([cx, cy, width, height].every((value) => Math.abs(value) <= 2)) {
            cx *= MODEL_INPUT_SIZE;
            cy *= MODEL_INPUT_SIZE;
            width *= MODEL_INPUT_SIZE;
            height *= MODEL_INPUT_SIZE;
        }

        const x1 = clamp(cx - width / 2, 0, MODEL_INPUT_SIZE);
        const y1 = clamp(cy - height / 2, 0, MODEL_INPUT_SIZE);
        const x2 = clamp(cx + width / 2, 0, MODEL_INPUT_SIZE);
        const y2 = clamp(cy + height / 2, 0, MODEL_INPUT_SIZE);
        if (x2 <= x1 || y2 <= y1) continue;

        const keypoints = [];
        for (let index = 0; index < KEYPOINT_COUNT; index += 1) {
            const offset = 5 + index * 3;
            let x = valueAt(candidate, offset);
            let y = valueAt(candidate, offset + 1);
            const pointConfidence = valueAt(candidate, offset + 2);

            if (Math.abs(x) <= 2 && Math.abs(y) <= 2) {
                x *= MODEL_INPUT_SIZE;
                y *= MODEL_INPUT_SIZE;
            }

            keypoints.push({
                x: clamp(x, 0, MODEL_INPUT_SIZE),
                y: clamp(y, 0, MODEL_INPUT_SIZE),
                confidence: Number.isFinite(pointConfidence) ? pointConfidence : 0
            });
        }

        people.push({ x1, y1, x2, y2, confidence, keypoints });
    }

    return nonMaxSuppression(people, NMS_THRESHOLD);
}

function nonMaxSuppression(boxes, threshold) {
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selected = [];

    while (sorted.length) {
        const current = sorted.shift();
        selected.push(current);

        for (let index = sorted.length - 1; index >= 0; index -= 1) {
            if (intersectionOverUnion(current, sorted[index]) > threshold) {
                sorted.splice(index, 1);
            }
        }
    }

    return selected;
}

function intersectionOverUnion(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    const union = areaA + areaB - intersection;
    return union > 0 ? intersection / union : 0;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
