"use strict";

importScripts("/vendor/onnxruntime/ort.all.min.js");

const MODEL_INPUT_SIZE = 640;
const CUSTOM_CONFIDENCE_THRESHOLD = 0.18;
const COCO_CONFIDENCE_THRESHOLD = 0.18;
const NMS_THRESHOLD = 0.45;
const COCO_PHONE_CLASS_ID = 67;

let session = null;
let inputName = null;
let outputName = null;
let cocoSession = null;
let cocoInputName = null;
let cocoOutputName = null;
let activeProvider = "wasm";

self.onmessage = async function (event) {
    const message = event.data || {};
    try {
        if (message.action === "INIT") {
            await initialize(message.modelUrl, message.cocoModelUrl);
        } else if (message.action === "INFERENCE") {
            await infer(message.tensorData, message.dims, message.meta);
        }
    } catch (error) {
        self.postMessage({
            status: "ERROR",
            error: error && error.message ? error.message : String(error)
        });
    }
};

async function initialize(modelUrl, cocoModelUrl) {
    if (!modelUrl) throw new Error("Missing phone model URL");
    if (!cocoModelUrl) throw new Error("Missing COCO phone model URL");

    ort.env.wasm.wasmPaths = "/vendor/onnxruntime/";
    ort.env.wasm.numThreads = Math.max(1, Math.min(3, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;
    ort.env.logLevel = "warning";
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";

    let customProvider = "wasm";
    let cocoProvider = "wasm";
    try {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all"
        });
        customProvider = "webgpu";
    } catch (error) {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all"
        });
    }

    inputName = session.inputNames[0];
    outputName = session.outputNames[0];

    try {
        cocoSession = await ort.InferenceSession.create(cocoModelUrl, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all"
        });
        cocoProvider = "webgpu";
    } catch (error) {
        cocoSession = await ort.InferenceSession.create(cocoModelUrl, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all"
        });
    }

    cocoInputName = cocoSession.inputNames[0];
    cocoOutputName = cocoSession.outputNames[0];
    activeProvider = customProvider === "webgpu" && cocoProvider === "webgpu"
        ? "webgpu"
        : "wasm";
    self.postMessage({
        status: activeProvider === "webgpu" ? "READY_WEBGPU" : "READY_WASM",
        modelCount: 2
    });
}

async function infer(tensorData, dims, meta) {
    if (!session || !cocoSession) throw new Error("Phone ensemble is not initialized");
    const startedAt = performance.now();
    const customFeeds = {};
    customFeeds[inputName] = new ort.Tensor("float32", tensorData, dims);
    const customOutputs = await session.run(customFeeds);
    const customOutput = customOutputs[outputName] || customOutputs[Object.keys(customOutputs)[0]];
    const customDetections = decodePhoneOutput(customOutput).map((box) => ({
        ...box,
        sourceCode: 0
    }));

    const cocoFeeds = {};
    cocoFeeds[cocoInputName] = new ort.Tensor("float32", tensorData, dims);
    const cocoOutputs = await cocoSession.run(cocoFeeds);
    const cocoOutput = cocoOutputs[cocoOutputName] || cocoOutputs[Object.keys(cocoOutputs)[0]];
    const cocoDetections = decodeCocoPhoneOutput(cocoOutput).map((box) => ({
        ...box,
        sourceCode: 1
    }));
    const detections = nonMaxSuppression(
        [...customDetections, ...cocoDetections],
        NMS_THRESHOLD
    );
    const flat = [];
    for (const box of detections) {
        flat.push(box.x1, box.y1, box.x2, box.y2, box.confidence, box.sourceCode);
    }
    self.postMessage({
        status: "RESULT",
        provider: activeProvider,
        inferenceMs: performance.now() - startedAt,
        meta: meta || { mode: "full" },
        counts: {
            custom: customDetections.length,
            coco: cocoDetections.length
        },
        data: flat
    });
}

function decodePhoneOutput(output) {
    const data = output.data;
    const dims = output.dims || [];
    if (dims.length !== 3 && dims.length !== 2) return [];

    let candidates;
    let attributes;
    let attributeFirst;
    if (dims.length === 3) {
        attributeFirst = dims[1] <= dims[2];
        attributes = attributeFirst ? dims[1] : dims[2];
        candidates = attributeFirst ? dims[2] : dims[1];
    } else {
        attributeFirst = false;
        candidates = dims[0];
        attributes = dims[1];
    }
    if (attributes < 5) return [];

    const valueAt = (candidate, attribute) => attributeFirst
        ? Number(data[attribute * candidates + candidate])
        : Number(data[candidate * attributes + attribute]);
    const boxes = [];
    for (let index = 0; index < candidates; index += 1) {
        const confidence = valueAt(index, 4);
        if (!Number.isFinite(confidence) || confidence < CUSTOM_CONFIDENCE_THRESHOLD) continue;

        let cx = valueAt(index, 0);
        let cy = valueAt(index, 1);
        let width = valueAt(index, 2);
        let height = valueAt(index, 3);
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
        const areaRatio = ((x2 - x1) * (y2 - y1)) / (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
        if (x2 <= x1 || y2 <= y1 || areaRatio < 0.00045 || areaRatio > 0.35) continue;
        boxes.push({ x1, y1, x2, y2, confidence });
    }

    return nonMaxSuppression(boxes, NMS_THRESHOLD);
}

function decodeCocoPhoneOutput(output) {
    const data = output.data;
    const dims = output.dims || [];
    if (dims.length !== 3 && dims.length !== 2) return [];

    let candidates;
    let attributes;
    let attributeFirst;
    if (dims.length === 3) {
        attributeFirst = dims[1] <= dims[2];
        attributes = attributeFirst ? dims[1] : dims[2];
        candidates = attributeFirst ? dims[2] : dims[1];
    } else {
        attributeFirst = false;
        candidates = dims[0];
        attributes = dims[1];
    }
    if (attributes < 4 + COCO_PHONE_CLASS_ID + 1) return [];

    const valueAt = (candidate, attribute) => attributeFirst
        ? Number(data[attribute * candidates + candidate])
        : Number(data[candidate * attributes + attribute]);
    const boxes = [];
    for (let index = 0; index < candidates; index += 1) {
        const confidence = valueAt(index, 4 + COCO_PHONE_CLASS_ID);
        if (!Number.isFinite(confidence) || confidence < COCO_CONFIDENCE_THRESHOLD) continue;

        let cx = valueAt(index, 0);
        let cy = valueAt(index, 1);
        let width = valueAt(index, 2);
        let height = valueAt(index, 3);
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
        const areaRatio = ((x2 - x1) * (y2 - y1)) / (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
        if (x2 <= x1 || y2 <= y1 || areaRatio < 0.00015 || areaRatio > 0.35) continue;
        boxes.push({ x1, y1, x2, y2, confidence });
    }
    return nonMaxSuppression(boxes, NMS_THRESHOLD);
}

function nonMaxSuppression(boxes, threshold) {
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selected = [];
    while (sorted.length) {
        const current = sorted.shift();
        selected.push(current);
        for (let index = sorted.length - 1; index >= 0; index -= 1) {
            if (iou(current, sorted[index]) > threshold) sorted.splice(index, 1);
        }
    }
    return selected;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return intersection / Math.max(1, areaA + areaB - intersection);
}

function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}
