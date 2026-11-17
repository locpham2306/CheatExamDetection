"use strict";

const MODEL_YOLO_URL = "/models/yolo26s.onnx";
const MODEL_PHONE_URL = "/models/best_phone_detection.onnx";
const MODEL_PHONE_COCO_URL = "/models/yolov8n-coco-phone.onnx";
const MODEL_FACE_URL = "/models/face_landmarker.task";
const MODEL_POSE_URL = "/models/yolov8n-pose.onnx";

const VIDEO_TARGET_WIDTH = 640;
const VIDEO_TARGET_HEIGHT = 480;
const YOLO_SIZE = 640;
const VIOLATION_COOLDOWN_MS = 3000;
const MAX_INCIDENT_CARDS = 100;

const YAW_THRESHOLD_DEGREES = 28;
const MAX_TRACKED_FACES = 5;
const POSE_KEYPOINT_CONFIDENCE = 0.15;
const POSE_HEAD_TURN_THRESHOLD = 0.38;
const POSE_RECORD_SIZE = 56;
const HEAD_TURN_ALERT_MS = 2000;
const HEAD_DIRECTION_GRACE_MS = 900;
const HEAD_CENTER_RESET_MS = 500;
// The dedicated YOLOv8s phone model is intentionally sampled less often than
// pose/person detection. Keep evidence alive across a slow CPU inference so
// two positive samples can confirm use without a one-frame alert.
const PHONE_ALERT_MS = 900;
const CLASSROOM_PHONE_ALERT_MS = 1800;
const PHONE_GRACE_MS = 2400;
const PERSON_TRACK_TTL_MS = 7000;
const DOCUMENT_TRACK_TTL_MS = 1600;
const DOCUMENT_OWNER_CONFIRM_MS = 650;
const PHONE_INFERENCE_INTERVAL_MS = 300;
const PHONE_TILE_RESULT_TTL_MS = 1800;

function computeYawFromMatrix(matrixObject) {
    if (!matrixObject || !matrixObject.data) return 0;
    const matrix = matrixObject.data;
    if (!matrix || matrix.length < 16) return 0;
    const m00 = Number(matrix[0]);
    const m10 = Number(matrix[4]);
    const m20 = Number(matrix[8]);
    if (![m00, m10, m20].every(Number.isFinite)) return 0;

    // Extract rotation around the vertical axis. The previous implementation
    // used atan2(m10, m00), which represents roll rather than left/right yaw.
    return Math.atan2(-m20, Math.hypot(m00, m10)) * (180 / Math.PI);
}

function getHeadDirection(yaw) {
    if (yaw <= -YAW_THRESHOLD_DEGREES) return "LEFT";
    if (yaw >= YAW_THRESHOLD_DEGREES) return "RIGHT";
    return "CENTER";
}

function buildFaceOrientations(faceLandmarks, transformationMatrices) {
    return (faceLandmarks || []).map((landmarks, index) => {
        const matrix = (transformationMatrices || [])[index];
        const yaw = computeYawFromMatrix(matrix);
        const xs = landmarks.map((point) => point.x).filter(Number.isFinite);
        const ys = landmarks.map((point) => point.y).filter(Number.isFinite);

        const minX = xs.length ? Math.min(...xs) : 0;
        const maxX = xs.length ? Math.max(...xs) : 0;
        const minY = ys.length ? Math.min(...ys) : 0;
        const maxY = ys.length ? Math.max(...ys) : 0;

        return {
            landmarks,
            yaw,
            direction: getHeadDirection(yaw),
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2
        };
    });
}

const video = document.getElementById("video");
const sourceCanvas = document.getElementById("sourceCanvas");
const sourceCtx = sourceCanvas.getContext("2d", { alpha: false });
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d", { alpha: true });

const yoloCanvas = document.getElementById("yoloCanvas");
const yoloCtx = yoloCanvas.getContext("2d", { willReadFrequently: true });

const rawCanvas = document.getElementById("rawCanvas");
const rawCtx = rawCanvas.getContext("2d", { willReadFrequently: true });

const snapshotCanvas = document.getElementById("snapshotCanvas");
const snapshotCtx = snapshotCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const stopVideoBtn = document.getElementById("stopVideoBtn");
const cameraSourceType = document.getElementById("cameraSourceType");
const cameraSelect = document.getElementById("cameraSelect");
const cameraQuality = document.getElementById("cameraQuality");
const refreshCamerasBtn = document.getElementById("refreshCamerasBtn");
const cameraStatus = document.getElementById("cameraStatus");
const cameraInfo = document.getElementById("cameraInfo");
const workspaceTitle = document.getElementById("workspaceTitle");
const liveSourcePanel = document.getElementById("liveSourcePanel");
const videoSourcePanel = document.getElementById("videoSourcePanel");
const workspaceModeButtons = document.querySelectorAll("[data-workspace-mode]");
const fallbackTitle = document.getElementById("fallbackTitle");
const fallbackHint = document.getElementById("fallbackHint");
const purgeBtn = document.getElementById("purgeBtn");
const videoFileInput = document.getElementById("videoFileInput");
const analyzeVideoBtn = document.getElementById("analyzeVideoBtn");
const selectedVideoName = document.getElementById("selectedVideoName");
const videoProgressWrap = document.getElementById("videoProgressWrap");
const videoProgress = document.getElementById("videoProgress");
const videoTime = document.getElementById("videoTime");

const fallback = document.getElementById("fallback");
const coreStatus = document.getElementById("coreStatus");
const yoloState = document.getElementById("yoloState");
const faceState = document.getElementById("faceState");
const phoneState = document.getElementById("phoneState");
const yawReadout = document.getElementById("yawReadout");
const incidentCount = document.getElementById("incidentCount");
const logFeed = document.getElementById("logFeed");
const emptyFeed = document.getElementById("emptyFeed");
const feedStatus = document.getElementById("feedStatus");
const sessionMeta = document.getElementById("sessionMeta");
const isolationBadge = document.getElementById("isolationBadge");
const exportZipBtn = document.getElementById("exportZipBtn");
const infoBtn = document.getElementById("infoBtn");
const infoModal = document.getElementById("infoModal");
const infoCloseBtn = document.getElementById("infoCloseBtn");
const gitRepoLink = document.getElementById("gitRepoLink");

let yoloWorker = null;
let phoneWorker = null;
let poseWorker = null;
let faceLandmarker = null;

let yoloReady = false;
let phoneReady = false;
let poseReady = false;
let faceReady = false;
let yoloBusy = false;
let phoneBusy = false;
let poseBusy = false;
let faceBusy = false;

let sessionActive = false;
let animationFrameId = null;
let mediaStream = null;
let sourceMode = null;
let selectedVideoFile = null;
let activeVideoUrl = null;
let cameraDevices = [];
let workspaceMode = "live";

let currentBoundingBoxes = [];
let currentPhoneBoxes = [];
let currentPosePeople = [];
let poseTileResults = { left: null, right: null };
let nextPoseTile = "left";
let currentFaceOrientations = [];
let currentYaw = 0;

let personTracks = new Map();
let documentTracks = new Map();
let nextPersonTrackId = 1;
let nextDocumentTrackId = 1;
let lastPhoneInferenceAt = 0;
let phoneTileResults = new Map();
let nextPhoneTileIndex = 0;

let lastViolationTimes = {};
let incidentTotal = 0;

const forbiddenLabels = {
    63: "Laptop / Secondary Screen",
    67: "Cell Phone"
};

const detectedLabels = {
    0: "Person",
    73: "Document / Book",
    ...forbiddenLabels
};

const directionLabels = {
    LEFT: "QUAY TRÁI",
    RIGHT: "QUAY PHẢI",
    CENTER: "NHÌN THẲNG",
    UNKNOWN: "CHƯA RÕ HƯỚNG"
};

document.addEventListener("DOMContentLoaded", bootSystem);
startBtn.addEventListener("click", startLiveSession);
stopBtn.addEventListener("click", terminateSession);
if (stopVideoBtn) stopVideoBtn.addEventListener("click", terminateSession);
if (refreshCamerasBtn) {
    refreshCamerasBtn.addEventListener("click", () => {
        refreshCameraDevices({ requestPermission: true });
    });
}
if (cameraSourceType) {
    cameraSourceType.addEventListener("change", () => {
        if (sessionActive) terminateSession();
        selectCameraForSourceType();
    });
}
workspaceModeButtons.forEach((button) => {
    button.addEventListener("click", () => setWorkspaceMode(button.dataset.workspaceMode));
});
videoFileInput.addEventListener("change", handleVideoSelection);
analyzeVideoBtn.addEventListener("click", startVideoAnalysis);
purgeBtn.addEventListener("click", purgeLogs);
exportZipBtn.addEventListener("click", exportViolationsAsZip);
if (infoBtn) infoBtn.addEventListener("click", openInfoModal);
if (infoCloseBtn) infoCloseBtn.addEventListener("click", closeInfoModal);
if (infoModal) infoModal.addEventListener("click", (event) => {
    if (event.target === infoModal) {
        closeInfoModal();
    }
});

function bootSystem() {
    updateIsolationBadge();
    initializeWorkers();
    initializePhoneWorker();
    initializePoseWorker();
    initializeFaceSystem();
    initializeInfoPanel();
    setWorkspaceMode("live", { stopActiveSession: false });
    refreshCameraDevices();

    if (navigator.mediaDevices?.addEventListener) {
        navigator.mediaDevices.addEventListener("devicechange", () => {
            refreshCameraDevices();
        });
    }
}

function setWorkspaceMode(mode, { stopActiveSession = true } = {}) {
    if (mode !== "live" && mode !== "video") return;
    if (stopActiveSession && sessionActive) terminateSession();

    workspaceMode = mode;
    const isLive = mode === "live";
    liveSourcePanel?.classList.toggle("hidden", !isLive);
    videoSourcePanel?.classList.toggle("hidden", isLive);
    workspaceModeButtons.forEach((button) => {
        const active = button.dataset.workspaceMode === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
    });

    if (workspaceTitle) {
        workspaceTitle.textContent = isLive ? "Camera trực tiếp" : "Phân tích video";
    }
    if (!sessionActive && fallbackTitle) {
        fallbackTitle.textContent = isLive ? "Camera chưa chạy" : "Chưa chọn video";
    }
    if (!sessionActive && fallbackHint) {
        fallbackHint.textContent = isLive
            ? "Chọn loại camera rồi bắt đầu giám sát."
            : "Chọn một video để bắt đầu phân tích.";
    }
}

const PHONE_CAMERA_PATTERN = /droidcam|camo|epoccam|iriun|iphone/i;

function isPhoneCamera(device) {
    return PHONE_CAMERA_PATTERN.test(device?.label || "");
}

function selectCameraForSourceType(preferredDeviceId = "") {
    if (!cameraSelect) return null;
    const wantsPhone = cameraSourceType?.value !== "computer";
    const labelsAvailable = cameraDevices.some((device) => device.label);
    const candidates = cameraDevices.filter((device) => {
        if (wantsPhone) return isPhoneCamera(device);
        return !labelsAvailable || !isPhoneCamera(device);
    });
    const selected = candidates.find((device) => device.deviceId === preferredDeviceId)
        || candidates[0]
        || null;

    cameraSelect.value = selected?.deviceId || "";
    if (!selected) {
        if (wantsPhone && !labelsAvailable) {
            setCameraStatus("Cần cấp quyền");
            if (cameraInfo) cameraInfo.textContent = "Nhấn Làm mới để trình duyệt nhận diện DroidCam.";
        } else {
            setCameraStatus("Không tìm thấy", "error");
            if (cameraInfo) {
                cameraInfo.textContent = wantsPhone
                    ? "Không thấy DroidCam. Kiểm tra DroidCam Client rồi nhấn Làm mới."
                    : "Không tìm thấy camera tích hợp hoặc webcam USB.";
            }
        }
        return null;
    }

    setCameraStatus(sessionActive && sourceMode === "camera" ? "Đang giám sát" : "Sẵn sàng",
        sessionActive && sourceMode === "camera" ? "online" : "");
    updateSelectedCameraHint();
    return selected;
}

function setCameraStatus(text, state = "") {
    if (!cameraStatus) return;
    cameraStatus.textContent = text;
    cameraStatus.classList.toggle("online", state === "online");
    cameraStatus.classList.toggle("error", state === "error");
}

function updateSelectedCameraHint() {
    if (!cameraInfo || !cameraSelect) return;
    const option = cameraSelect.selectedOptions?.[0];
    const label = option?.textContent || "Chưa chọn camera";
    const wantsPhone = cameraSourceType?.value !== "computer";
    cameraInfo.textContent = wantsPhone
        ? `${label} · camera điện thoại đã sẵn sàng`
        : `${label} · camera máy tính đã sẵn sàng`;
}

async function refreshCameraDevices({ requestPermission = false } = {}) {
    if (!navigator.mediaDevices?.enumerateDevices) {
        setCameraStatus("Không hỗ trợ", "error");
        if (cameraInfo) cameraInfo.textContent = "Trình duyệt này không hỗ trợ chọn thiết bị camera.";
        return [];
    }

    const previousDeviceId = cameraSelect?.value || "";
    let permissionStream = null;
    if (refreshCamerasBtn) refreshCamerasBtn.disabled = true;

    try {
        if (requestPermission && !mediaStream) {
            permissionStream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: true
            });
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraDevices = devices.filter((device) => device.kind === "videoinput");

        if (!cameraSelect) return cameraDevices;
        cameraSelect.replaceChildren();

        if (!cameraDevices.length) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "Không tìm thấy camera";
            cameraSelect.append(option);
            setCameraStatus("Không có camera", "error");
            if (cameraInfo) cameraInfo.textContent = "Bật DroidCam rồi nhấn Làm mới.";
            return cameraDevices;
        }

        cameraDevices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.append(option);
        });

        selectCameraForSourceType(previousDeviceId);
        return cameraDevices;
    } catch (error) {
        console.error("Unable to enumerate cameras:", error);
        setCameraStatus("Chưa cấp quyền", "error");
        if (cameraInfo) cameraInfo.textContent = formatCameraError(error);
        return [];
    } finally {
        if (permissionStream) {
            permissionStream.getTracks().forEach((track) => track.stop());
        }
        if (refreshCamerasBtn) refreshCamerasBtn.disabled = sessionActive;
    }
}

function formatCameraError(error) {
    if (error?.name === "NotAllowedError") {
        return "Quyền camera đang bị chặn. Hãy cho phép camera trong trình duyệt rồi thử lại.";
    }
    if (error?.name === "NotFoundError") {
        return "Không tìm thấy camera. Kiểm tra DroidCam Client đang hoạt động rồi nhấn Làm mới.";
    }
    if (error?.name === "NotReadableError") {
        return "Camera đang được ứng dụng khác sử dụng. Hãy đóng ứng dụng đó rồi thử lại.";
    }
    if (error?.name === "OverconstrainedError") {
        return "DroidCam không hỗ trợ chất lượng đã chọn. Hãy thử 720p hoặc 480p.";
    }
    return error?.message || "Không mở được camera.";
}

function initializeInfoPanel() {
    if (gitRepoLink) {
        gitRepoLink.href = "https://github.com/ChinmoyDeb/ai-exam-proctor";
        gitRepoLink.textContent = "https://github.com/ChinmoyDeb/ai-exam-proctor";
    }
}

function openInfoModal() {
    if (!infoModal) return;
    infoModal.classList.remove("hidden");
    document.body.style.overflow = "hidden";
}

function closeInfoModal() {
    if (!infoModal) return;
    infoModal.classList.add("hidden");
    document.body.style.overflow = "";
}

function updateIsolationBadge() {
    if (self.crossOriginIsolated) {
        isolationBadge.textContent = "Isolation: true";
    } else {
        isolationBadge.textContent = "Isolation: false";
        isolationBadge.style.color = "#fca5a5";
        isolationBadge.style.borderColor = "rgba(248,113,113,0.35)";
        isolationBadge.style.background = "rgba(239,68,68,0.12)";
    }
}

function initializeWorkers() {
    const epoch = Date.now();

    yoloWorker = new Worker(`/workers/yoloWorker.js?t=${epoch}`);

    yoloWorker.onmessage = handleYoloMessage;
    yoloWorker.onerror = handleYoloError;

    yoloWorker.postMessage({
        action: "INIT",
        modelUrl: MODEL_YOLO_URL
    });

    if (coreStatus) coreStatus.textContent = "Edge Core Initializing";
}

function initializePhoneWorker() {
    const epoch = Date.now();
    phoneReady = false;
    phoneBusy = false;

    if (phoneWorker) {
        try {
            phoneWorker.terminate();
        } catch (error) { }
    }

    phoneWorker = new Worker(`/workers/phoneWorker.js?t=${epoch}`);
    phoneWorker.onmessage = handlePhoneMessage;
    phoneWorker.onerror = handlePhoneError;
    phoneWorker.postMessage({
        action: "INIT",
        modelUrl: MODEL_PHONE_URL,
        cocoModelUrl: MODEL_PHONE_COCO_URL
    });
}

function handlePhoneMessage(event) {
    const message = event.data || {};

    if (message.status === "READY_WEBGPU" || message.status === "READY_WASM") {
        phoneReady = true;
        if (phoneState) {
            const provider = message.status === "READY_WEBGPU" ? "GPU" : "CPU";
            phoneState.textContent = `${message.modelCount || 1} MODEL · ${provider} READY`;
        }
        if (yoloState && yoloReady) yoloState.textContent = "READY + PHONE";
        return;
    }

    if (message.status === "RESULT") {
        phoneBusy = false;
        if (!sessionActive) {
            currentPhoneBoxes = [];
            return;
        }
        const meta = message.meta || { mode: "full" };
        const normalized = normalizePhoneResult(message.data || []);
        if (meta.mode === "tile") {
            const tileBoxes = transformPhoneBoxes(normalized, meta).map((box) => ({
                ...box,
                detector: `${box.detector}-tile`
            }));
            phoneTileResults.set(meta.tile, {
                boxes: tileBoxes,
                updatedAt: performance.now()
            });
            const recentBoxes = [];
            const now = performance.now();
            for (const [tile, result] of phoneTileResults.entries()) {
                if (now - result.updatedAt > PHONE_TILE_RESULT_TTL_MS) {
                    phoneTileResults.delete(tile);
                    continue;
                }
                recentBoxes.push(...result.boxes);
            }
            currentPhoneBoxes = dedupePhoneBoxes(recentBoxes);
        } else {
            phoneTileResults.clear();
            currentPhoneBoxes = normalized;
        }
        if (phoneState) {
            const provider = message.provider === "webgpu" ? "GPU" : "CPU";
            const latency = Number.isFinite(message.inferenceMs)
                ? `${Math.round(message.inferenceMs)} ms`
                : "READY";
            const counts = message.counts || {};
            phoneState.textContent = `2 MODEL · ${provider} · ${latency} · ${counts.custom || 0}+${counts.coco || 0}`;
        }
        updateBehaviorTracking({ updateDirection: false, updateObjects: true });
        return;
    }

    if (message.status === "ERROR") {
        phoneBusy = false;
        phoneReady = false;
        currentPhoneBoxes = [];
        phoneTileResults.clear();
        if (phoneState) phoneState.textContent = "ERROR";
        console.error("Phone worker error:", message.error);
        setTimeout(initializePhoneWorker, 1500);
    }
}

function handlePhoneError(error) {
    phoneBusy = false;
    phoneReady = false;
    currentPhoneBoxes = [];
    phoneTileResults.clear();
    if (phoneState) phoneState.textContent = "ERROR";
    console.error("Phone worker crashed:", error);
    setTimeout(initializePhoneWorker, 1500);
}

function initializePoseWorker() {
    const epoch = Date.now();
    poseReady = false;
    poseBusy = false;
    updateHeadModelState();

    if (poseWorker) {
        try {
            poseWorker.terminate();
        } catch (error) { }
    }

    poseWorker = new Worker(`/workers/poseWorker.js?t=${epoch}`);
    poseWorker.onmessage = handlePoseMessage;
    poseWorker.onerror = handlePoseError;
    poseWorker.postMessage({ action: "INIT", modelUrl: MODEL_POSE_URL });
}

function handlePoseMessage(event) {
    const message = event.data || {};

    if (message.status === "READY_WEBGPU" || message.status === "READY_WASM") {
        poseReady = true;
        updateHeadModelState();
        unlockStartIfReady();
        return;
    }

    if (message.status === "RESULT") {
        poseBusy = false;
        updateHeadModelState();

        if (!sessionActive) {
            currentPosePeople = [];
            poseTileResults = { left: null, right: null };
            return;
        }

        const normalizedPeople = normalizePoseResult(message.data || []);
        const meta = message.meta || null;
        if (meta && meta.mode === "tile") {
            const transformed = transformPosePeople(normalizedPeople, meta);
            poseTileResults[meta.tile] = {
                people: transformed,
                receivedAt: performance.now()
            };
            const freshTiles = Object.values(poseTileResults)
                .filter((entry) => entry && performance.now() - entry.receivedAt < 3500)
                .flatMap((entry) => entry.people);
            currentPosePeople = dedupePosePeople(freshTiles);
        } else {
            poseTileResults = { left: null, right: null };
            currentPosePeople = normalizedPeople;
        }
        updateBehaviorTracking({ updateDirection: true, updateObjects: false });
        if (sourceMode === "video" && yawReadout) {
            const headCount = currentPosePeople.filter((person) => person.headDetected).length;
            yawReadout.textContent = `${headCount}/${currentPosePeople.length} có keypoint`;
        }
        return;
    }

    if (message.status === "ERROR") {
        poseBusy = false;
        poseReady = false;
        updateHeadModelState();
        console.error("Pose worker error:", message.error);
        setTimeout(initializePoseWorker, 1200);
    }
}

function handlePoseError(error) {
    poseBusy = false;
    poseReady = false;
    updateHeadModelState();
    console.error("Pose worker crashed:", error);
    setTimeout(initializePoseWorker, 1200);
}

function updateHeadModelState() {
    if (!faceState) return;
    if (!poseReady) {
        faceState.textContent = "POSE LOADING";
        return;
    }
    if (poseBusy) {
        faceState.textContent = "POSE INFERENCE";
        return;
    }
    faceState.textContent = faceReady ? "POSE + FACE" : "POSE READY";
}

async function initializeFaceSystem() {
    updateHeadModelState();
    try {
        const vision = await import(`/vendor/mediapipe/vision_bundle.mjs?t=${Date.now()}`);
        const { FaceLandmarker, FilesetResolver } = vision;

        if (!FaceLandmarker || !FilesetResolver) {
            throw new Error("MediaPipe bundle missing FaceLandmarker or FilesetResolver");
        }

        const visionFileset = await FilesetResolver.forVisionTasks("/vendor/mediapipe/wasm");

        faceLandmarker = await FaceLandmarker.createFromOptions(visionFileset, {
            baseOptions: {
                modelAssetPath: MODEL_FACE_URL,
                delegate: "CPU"
            },
            runningMode: "VIDEO",
            numFaces: MAX_TRACKED_FACES,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true
        });

        faceReady = true;
        updateHeadModelState();
        unlockStartIfReady();
    } catch (err) {
        setFaceSystemError(err);
    }
}

function setFaceSystemError(err) {
    const message = err && err.message ? err.message : String(err);
    faceReady = false;
    faceBusy = false;
    updateHeadModelState();
    console.warn("Optional close-range face system error:", message);
    setTimeout(() => {
        console.info("Retrying face system initialization...");
        initializeFaceSystem();
    }, 2000);
}

function handleYoloMessage(event) {
    const message = event.data || {};

    if (message.status === "READY_WEBGPU") {
        yoloReady = true;
        if (yoloState) yoloState.textContent = "READY_WEBGPU";
        unlockStartIfReady();
        return;
    }

    if (message.status === "READY_WASM") {
        yoloReady = true;
        if (yoloState) yoloState.textContent = "READY_WASM";
        unlockStartIfReady();
        return;
    }

    if (message.status === "RESULT") {
        yoloBusy = false;
        if (yoloState) yoloState.textContent = yoloReady ? "READY" : "INIT";

        if (!sessionActive) {
            currentBoundingBoxes = [];
            return;
        }

        currentBoundingBoxes = normalizeYoloResult(message.data || []);
        updateBehaviorTracking({ updateDirection: false, updateObjects: true });

        return;
    }

    if (message.status === "ERROR") {
        yoloBusy = false;
        if (yoloState) yoloState.textContent = "ERROR";
        console.error("YOLO worker error:", message.error);
        try {
            if (yoloWorker) {
                yoloWorker.terminate();
            }
        } catch (e) { }

        setTimeout(() => {
            console.info("Reinitializing YOLO worker after error...");
            initializeWorkers();
        }, 1000);
    }
}

function handleFaceMessage(event) {
    const message = event.data || {};

    if (message.status === "READY") {
        faceReady = true;
        updateHeadModelState();
        unlockStartIfReady();
        return;
    }

    if (message.status === "RESULT") {
        faceBusy = false;
        updateHeadModelState();

        const data = message.data || {};
        currentFaceOrientations = buildFaceOrientations(
            data.faceLandmarks || [],
            data.facialTransformationMatrixes || []
        );
        currentYaw = Number(data.yaw || 0);
        if (yawReadout) yawReadout.textContent = `${currentYaw.toFixed(2)}°`;

        // Do not create immediate face/no-face incidents here. Classroom
        // cameras often cannot resolve every face in every frame. Sustained
        // head-turn alerts are emitted by the tracked-person behavior engine.

        return;
    }

    if (message.status === "ERROR") {
        faceBusy = false;
        faceReady = false;
        updateHeadModelState();
        console.error("Face worker error:", message.error);
    }
}

function handleYoloError(error) {
    yoloBusy = false;
    if (yoloState) yoloState.textContent = "ERROR";
    console.error("YOLO worker crashed:", error);
    try {
        if (yoloWorker) {
            yoloWorker.terminate();
        }
    } catch (e) { }

    setTimeout(() => {
        console.info("Reinitializing YOLO worker after crash...");
        initializeWorkers();
    }, 1000);
}

function handleFaceError(error) {
    faceBusy = false;
    faceReady = false;
    updateHeadModelState();
    console.error("Face worker crashed:", error);
}

function unlockStartIfReady() {
    const modelsReady = yoloReady && poseReady;
    if (startBtn) startBtn.disabled = !modelsReady || sessionActive;
    if (analyzeVideoBtn) {
        analyzeVideoBtn.disabled = !modelsReady || !selectedVideoFile || sessionActive;
    }

    if (modelsReady) {
        if (coreStatus) coreStatus.textContent = "Edge Core Active";
    }
}

function handleVideoSelection() {
    const file = videoFileInput.files && videoFileInput.files[0];
    selectedVideoFile = file || null;

    if (selectedVideoName) {
        selectedVideoName.textContent = file
            ? `${file.name} · ${formatFileSize(file.size)}`
            : "Chưa chọn video";
        selectedVideoName.style.color = "";
    }

    unlockStartIfReady();
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const unitIndex = Math.min(
        Math.floor(Math.log(bytes) / Math.log(1024)),
        units.length - 1
    );
    const value = bytes / (1024 ** unitIndex);
    return `${value.toFixed(unitIndex > 1 ? 1 : 0)} ${units[unitIndex]}`;
}

async function startLiveSession() {
    if (sessionActive) return;

    try {
        releaseActiveVideoUrl();
        sourceMode = "camera";

        const labelsAvailable = cameraDevices.some((device) => device.label);
        if (!cameraDevices.length || !labelsAvailable) {
            await refreshCameraDevices({ requestPermission: true });
        }

        const selectedCamera = selectCameraForSourceType(cameraSelect?.value || "");
        if (!selectedCamera) {
            const cameraName = cameraSourceType?.value === "computer"
                ? "camera máy tính"
                : "DroidCam";
            throw new DOMException(`Không tìm thấy ${cameraName}`, "NotFoundError");
        }

        const [requestedWidth, requestedHeight] = (cameraQuality?.value || "1280x720")
            .split("x")
            .map(Number);
        const selectedDeviceId = cameraSelect?.value || "";
        const videoConstraints = {
            width: { ideal: requestedWidth || 1280 },
            height: { ideal: requestedHeight || 720 },
            frameRate: { ideal: 30, max: 30 }
        };
        if (selectedDeviceId) {
            videoConstraints.deviceId = { exact: selectedDeviceId };
        }

        setCameraStatus("Đang kết nối...");
        if (cameraInfo) cameraInfo.textContent = "Đang mở luồng hình ảnh từ thiết bị đã chọn.";
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: videoConstraints
        });

        video.srcObject = mediaStream;
        await waitForVideoMetadata();
        await video.play();

        const videoTrack = mediaStream.getVideoTracks()[0];
        const settings = videoTrack?.getSettings?.() || {};
        const selectedLabel = videoTrack?.label || cameraSelect?.selectedOptions?.[0]?.textContent || "Camera";
        const actualWidth = settings.width || video.videoWidth;
        const actualHeight = settings.height || video.videoHeight;
        const actualFps = settings.frameRate ? `${Math.round(settings.frameRate)} FPS` : "FPS tự động";

        setCameraStatus("Đang giám sát", "online");
        if (cameraInfo) {
            const mirrorStatus = cameraSourceType?.value === "computer" ? " · đã sửa chiều webcam" : "";
            cameraInfo.textContent = `${selectedLabel} · ${actualWidth}×${actualHeight} · ${actualFps}${mirrorStatus}`;
        }
        if (videoTrack) {
            videoTrack.addEventListener("ended", () => {
                if (sessionActive && sourceMode === "camera") {
                    setCameraStatus("Mất kết nối", "error");
                    if (cameraInfo) cameraInfo.textContent = "Luồng camera đã dừng. Kiểm tra thiết bị rồi kết nối lại.";
                    terminateSession();
                }
            }, { once: true });
        }

        activateProcessingSession(
            `${actualWidth}×${actualHeight} · ${selectedLabel}`
        );
    } catch (error) {
        console.error("Camera initialization failed:", error);
        resetFailedSource();
        setCameraStatus("Kết nối thất bại", "error");
        if (cameraInfo) cameraInfo.textContent = formatCameraError(error);
    }
}

async function startVideoAnalysis() {
    if (sessionActive || !selectedVideoFile || !yoloReady || !poseReady) return;

    try {
        sourceMode = "video";
        video.srcObject = null;
        releaseActiveVideoUrl();
        activeVideoUrl = URL.createObjectURL(selectedVideoFile);
        video.src = activeVideoUrl;
        video.muted = true;
        video.load();

        await waitForVideoMetadata();
        await video.play();

        video.onended = terminateSession;
        activateProcessingSession(
            `${video.videoWidth}×${video.videoHeight} · ${selectedVideoFile.name}`
        );
    } catch (error) {
        console.error("Video initialization failed:", error);
        if (selectedVideoName) {
            selectedVideoName.textContent = "Không đọc được video. Hãy thử MP4 hoặc WebM.";
            selectedVideoName.style.color = "#dc2626";
        }
        resetFailedSource();
    }
}

function waitForVideoMetadata() {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
        const handleLoaded = () => {
            cleanup();
            resolve();
        };
        const handleError = () => {
            cleanup();
            reject(video.error || new Error("Trình duyệt không đọc được video"));
        };
        const cleanup = () => {
            video.removeEventListener("loadedmetadata", handleLoaded);
            video.removeEventListener("error", handleError);
        };

        video.addEventListener("loadedmetadata", handleLoaded, { once: true });
        video.addEventListener("error", handleError, { once: true });
    });
}

function activateProcessingSession(metaText) {
    resetBehaviorTracking();
    poseTileResults = { left: null, right: null };
    nextPoseTile = "left";
    sessionActive = true;
    fallback.classList.add("hidden");
    const isCameraSession = sourceMode === "camera";
    startBtn.classList.toggle("hidden", isCameraSession);
    stopBtn.classList.toggle("hidden", !isCameraSession);
    if (stopVideoBtn) stopVideoBtn.classList.toggle("hidden", isCameraSession);
    videoFileInput.disabled = true;
    analyzeVideoBtn.disabled = true;
    if (cameraSourceType) cameraSourceType.disabled = true;
    if (cameraSelect) cameraSelect.disabled = true;
    if (cameraQuality) cameraQuality.disabled = true;
    if (refreshCamerasBtn) refreshCamerasBtn.disabled = true;

    if (videoProgressWrap) {
        videoProgressWrap.classList.toggle("hidden", sourceMode !== "video");
    }

    resizeCanvases();
    sessionMeta.textContent = metaText;
    updateVideoProgress();
    animationFrameId = requestAnimationFrame(processFrameLoop);
}

function terminateSession() {
    const endedSourceMode = sourceMode;
    sessionActive = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (mediaStream) {
        for (const track of mediaStream.getTracks()) {
            track.stop();
        }

        mediaStream = null;
    }

    video.pause();
    video.onended = null;
    video.srcObject = null;
    releaseActiveVideoUrl();
    sourceMode = null;

    fallback.classList.remove("hidden");
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    if (stopVideoBtn) stopVideoBtn.classList.add("hidden");
    videoFileInput.disabled = false;
    if (cameraSourceType) cameraSourceType.disabled = false;
    if (cameraSelect) cameraSelect.disabled = false;
    if (cameraQuality) cameraQuality.disabled = false;
    if (refreshCamerasBtn) refreshCamerasBtn.disabled = false;
    if (videoProgressWrap) videoProgressWrap.classList.add("hidden");

    currentBoundingBoxes = [];
    currentPhoneBoxes = [];
    currentPosePeople = [];
    poseTileResults = { left: null, right: null };
    nextPoseTile = "left";
    currentFaceOrientations = [];
    currentYaw = 0;
    lastPhoneInferenceAt = 0;
    resetBehaviorTracking();

    yawReadout.textContent = "0/0";
    sessionMeta.textContent = "Session Pipeline Disengaged";

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    updateVideoProgress();
    unlockStartIfReady();
    setWorkspaceMode(workspaceMode, { stopActiveSession: false });

    if (endedSourceMode === "camera") {
        setCameraStatus("Đã dừng");
        updateSelectedCameraHint();
    }
}

function releaseActiveVideoUrl() {
    if (!activeVideoUrl) return;
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(activeVideoUrl);
    activeVideoUrl = null;
}

function resetFailedSource() {
    if (mediaStream) {
        for (const track of mediaStream.getTracks()) track.stop();
        mediaStream = null;
    }
    video.srcObject = null;
    releaseActiveVideoUrl();
    sourceMode = null;
    videoFileInput.disabled = false;
    if (cameraSourceType) cameraSourceType.disabled = false;
    if (cameraSelect) cameraSelect.disabled = false;
    if (cameraQuality) cameraQuality.disabled = false;
    if (refreshCamerasBtn) refreshCamerasBtn.disabled = false;
    unlockStartIfReady();
}

function resizeCanvases() {
    const width = video.videoWidth || VIDEO_TARGET_WIDTH;
    const height = video.videoHeight || VIDEO_TARGET_HEIGHT;

    sourceCanvas.width = width;
    sourceCanvas.height = height;

    overlayCanvas.width = width;
    overlayCanvas.height = height;

    rawCanvas.width = width;
    rawCanvas.height = height;

    snapshotCanvas.width = width;
    snapshotCanvas.height = height;

    yoloCanvas.width = YOLO_SIZE;
    yoloCanvas.height = YOLO_SIZE;
}

function processFrameLoop(timestamp) {
    if (!sessionActive) return;

    renderSourceFrame();
    drawOverlay();
    updateVideoProgress();

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        if ((yoloReady && !yoloBusy) || (poseReady && !poseBusy)
            || (phoneReady && !phoneBusy)) {
            dispatchModelInference();
        }

        if (sourceMode === "camera" && faceReady && !faceBusy) {
            dispatchFaceInference(timestamp);
        }
    }

    animationFrameId = requestAnimationFrame(processFrameLoop);
}

function renderSourceFrame() {
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const correctComputerWebcamMirror = sourceMode === "camera"
        && cameraSourceType?.value === "computer";

    sourceCtx.save();
    sourceCtx.setTransform(1, 0, 0, 1, 0, 0);
    sourceCtx.fillStyle = "#020617";
    sourceCtx.fillRect(0, 0, width, height);
    if (correctComputerWebcamMirror) {
        sourceCtx.translate(width, 0);
        sourceCtx.scale(-1, 1);
    }
    sourceCtx.drawImage(video, 0, 0, width, height);
    sourceCtx.restore();
}

function updateVideoProgress() {
    if (!videoProgress || !videoTime) return;

    if (sourceMode !== "video") {
        videoProgress.max = 1;
        videoProgress.value = 0;
        videoTime.textContent = "00:00 / 00:00";
        return;
    }

    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const currentTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    videoProgress.max = Math.max(duration, 1);
    videoProgress.value = Math.min(currentTime, videoProgress.max);
    videoTime.textContent = `${formatDuration(currentTime)} / ${formatDuration(duration)}`;
}

function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    const minuteText = String(minutes).padStart(hours ? 2 : 1, "0");
    const secondText = String(remainder).padStart(2, "0");
    return hours
        ? `${hours}:${minuteText}:${secondText}`
        : `${minuteText.padStart(2, "0")}:${secondText}`;
}

function dispatchModelInference() {
    const runYolo = yoloReady && !yoloBusy;
    const runPose = poseReady && !poseBusy;
    const now = performance.now();
    const runPhone = phoneReady && !phoneBusy
        && now - lastPhoneInferenceAt >= PHONE_INFERENCE_INTERVAL_MS;
    if (!runYolo && !runPose && !runPhone) return;

    try {
        yoloCtx.drawImage(sourceCanvas, 0, 0, YOLO_SIZE, YOLO_SIZE);
        const imageData = yoloCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);
        const tensorData = imageDataToCHWTensor(imageData);
        const usePoseTiles = runPose && shouldUsePoseTiling();
        const yoloTensor = runYolo ? tensorData : null;
        let poseTensor = null;
        let poseMeta = { mode: "full" };

        if (runPose && usePoseTiles) {
            poseMeta = getNextPoseTile();
            const sourceX = Math.round(video.videoWidth * poseMeta.xStart);
            const sourceWidth = Math.round(video.videoWidth * poseMeta.xScale);
            yoloCtx.drawImage(
                sourceCanvas,
                sourceX,
                0,
                sourceWidth,
                video.videoHeight,
                0,
                0,
                YOLO_SIZE,
                YOLO_SIZE
            );
            poseTensor = imageDataToCHWTensor(
                yoloCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE)
            );
        } else if (runPose) {
            poseTensor = runYolo ? tensorData.slice() : tensorData;
        }
        let phoneTensor = null;
        let phoneMeta = { mode: "full" };
        if (runPhone && shouldUsePhoneTiling()) {
            phoneMeta = getNextPhoneTile();
            const sourceX = Math.round(video.videoWidth * phoneMeta.xStart);
            const sourceY = Math.round(video.videoHeight * phoneMeta.yStart);
            const sourceWidth = Math.round(video.videoWidth * phoneMeta.xScale);
            const sourceHeight = Math.round(video.videoHeight * phoneMeta.yScale);
            yoloCtx.drawImage(
                sourceCanvas,
                sourceX,
                sourceY,
                sourceWidth,
                sourceHeight,
                0,
                0,
                YOLO_SIZE,
                YOLO_SIZE
            );
            phoneTensor = imageDataToCHWTensor(
                yoloCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE)
            );
        } else if (runPhone) {
            phoneTensor = (runYolo || (runPose && !usePoseTiles))
                ? tensorData.slice()
                : tensorData;
        }

        if (runYolo) {
            yoloBusy = true;
            if (yoloState) yoloState.textContent = "INFERENCE";
            yoloWorker.postMessage(
                {
                    action: "INFERENCE",
                    tensorData: yoloTensor,
                    dims: [1, 3, YOLO_SIZE, YOLO_SIZE]
                },
                [yoloTensor.buffer]
            );
        }

        if (runPose) {
            poseBusy = true;
            updateHeadModelState();
            poseWorker.postMessage(
                {
                    action: "INFERENCE",
                    tensorData: poseTensor,
                    dims: [1, 3, YOLO_SIZE, YOLO_SIZE],
                    meta: poseMeta
                },
                [poseTensor.buffer]
            );
        }

        if (runPhone) {
            phoneBusy = true;
            lastPhoneInferenceAt = now;
            phoneWorker.postMessage(
                {
                    action: "INFERENCE",
                    tensorData: phoneTensor,
                    dims: [1, 3, YOLO_SIZE, YOLO_SIZE],
                    meta: phoneMeta
                },
                [phoneTensor.buffer]
            );
        }
    } catch (error) {
        if (runYolo) yoloBusy = false;
        if (runPose) poseBusy = false;
        if (runPhone) phoneBusy = false;
        updateHeadModelState();
        console.error("Model dispatch failed:", error);
    }
}

function dispatchFaceInference(timestamp) {
    if (!faceLandmarker) {
        faceBusy = false;
        return;
    }

    (async () => {
        try {
            faceBusy = true;

            rawCtx.drawImage(sourceCanvas, 0, 0, rawCanvas.width, rawCanvas.height);

            const safeTimestamp = Math.round(timestamp || performance.now());

            const result = await faceLandmarker.detectForVideo(rawCanvas, safeTimestamp);

            faceBusy = false;
            updateHeadModelState();

            if (!sessionActive) return;

            const faceLandmarks = result.faceLandmarks || [];
            const facialTransformationMatrixes = result.facialTransformationMatrixes || [];

            currentFaceOrientations = buildFaceOrientations(
                faceLandmarks,
                facialTransformationMatrixes
            );
            updateBehaviorTracking({ updateDirection: true, updateObjects: false });

            let yaw = 0;
            if (currentFaceOrientations.length) {
                yaw = currentFaceOrientations[0].yaw;
            }

            currentYaw = Number(yaw || 0);
            if (yawReadout) yawReadout.textContent = `${currentYaw.toFixed(2)}°`;

            // Head turns are emitted by updateBehaviorTracking only after the
            // same tracked person remains left/right for two seconds.
        } catch (error) {
            faceBusy = false;
            setFaceSystemError(error);
        }
    })();
}

function imageDataToCHWTensor(imageData) {
    const { data, width, height } = imageData;
    const planeSize = width * height;
    const tensor = new Float32Array(3 * planeSize);

    for (let i = 0, pixel = 0; i < data.length; i += 4, pixel++) {
        tensor[pixel] = data[i] / 255.0;
        tensor[planeSize + pixel] = data[i + 1] / 255.0;
        tensor[planeSize * 2 + pixel] = data[i + 2] / 255.0;
    }

    return tensor;
}

function normalizeYoloResult(flatData) {
    const boxes = [];

    for (let i = 0; i + 5 < flatData.length; i += 6) {
        const x1 = Number(flatData[i]);
        const y1 = Number(flatData[i + 1]);
        const x2 = Number(flatData[i + 2]);
        const y2 = Number(flatData[i + 3]);
        const confidence = Number(flatData[i + 4]);
        const classId = Number(flatData[i + 5]);

        if (!Number.isFinite(confidence) || confidence < 0.25) {
            continue;
        }

        const label = detectedLabels[classId] || "Detected Object";
        const violation = forbiddenLabels[classId] !== undefined;

        boxes.push({
            x1,
            y1,
            x2,
            y2,
            confidence,
            classId,
            label,
            violation
        });
    }

    return boxes;
}

function normalizePhoneResult(flatData) {
    const boxes = [];

    for (let i = 0; i + 5 < flatData.length; i += 6) {
        const x1 = Number(flatData[i]);
        const y1 = Number(flatData[i + 1]);
        const x2 = Number(flatData[i + 2]);
        const y2 = Number(flatData[i + 3]);
        const confidence = Number(flatData[i + 4]);
        const sourceCode = Number(flatData[i + 5]);
        if (!Number.isFinite(confidence) || confidence < 0.18) continue;
        boxes.push({
            x1,
            y1,
            x2,
            y2,
            confidence,
            classId: 67,
            label: "Cell Phone",
            violation: true,
            detector: sourceCode === 1 ? "phone-coco" : "phone-custom"
        });
    }

    return boxes;
}

function shouldUsePhoneTiling() {
    if (sourceMode === "video") return true;
    return getDisplayedPersonCount() >= 3;
}

function getNextPhoneTile() {
    const tiles = [
        { tile: "top-left", xStart: 0, yStart: 0, xScale: 0.56, yScale: 0.58 },
        { tile: "top-right", xStart: 0.44, yStart: 0, xScale: 0.56, yScale: 0.58 },
        { tile: "bottom-left", xStart: 0, yStart: 0.42, xScale: 0.56, yScale: 0.58 },
        { tile: "bottom-right", xStart: 0.44, yStart: 0.42, xScale: 0.56, yScale: 0.58 }
    ];
    const tile = tiles[nextPhoneTileIndex % tiles.length];
    nextPhoneTileIndex = (nextPhoneTileIndex + 1) % tiles.length;
    return { mode: "tile", ...tile };
}

function transformPhoneBoxes(boxes, meta) {
    const transformX = (value) => (meta.xStart + (value / YOLO_SIZE) * meta.xScale) * YOLO_SIZE;
    const transformY = (value) => (meta.yStart + (value / YOLO_SIZE) * meta.yScale) * YOLO_SIZE;
    return boxes.map((box) => ({
        ...box,
        x1: transformX(box.x1),
        y1: transformY(box.y1),
        x2: transformX(box.x2),
        y2: transformY(box.y2)
    }));
}

function dedupePhoneBoxes(boxes) {
    const selected = [];
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    for (const box of sorted) {
        if (selected.some((candidate) => boxIou(candidate, box) > 0.42)) continue;
        selected.push(box);
    }
    return selected;
}

function normalizePoseResult(flatData) {
    const people = [];

    for (let offset = 0; offset + POSE_RECORD_SIZE <= flatData.length; offset += POSE_RECORD_SIZE) {
        const confidence = Number(flatData[offset + 4]);
        if (!Number.isFinite(confidence) || confidence < 0.25) continue;

        const keypoints = [];
        for (let index = 0; index < 17; index += 1) {
            const pointOffset = offset + 5 + index * 3;
            keypoints.push({
                x: Number(flatData[pointOffset]),
                y: Number(flatData[pointOffset + 1]),
                confidence: Number(flatData[pointOffset + 2])
            });
        }

        const head = estimatePoseHeadDirection(keypoints);
        people.push({
            x1: Number(flatData[offset]),
            y1: Number(flatData[offset + 1]),
            x2: Number(flatData[offset + 2]),
            y2: Number(flatData[offset + 3]),
            confidence,
            keypoints,
            direction: head.direction,
            headDetected: head.headDetected
        });
    }

    return people;
}

function shouldUsePoseTiling() {
    const aspectRatio = video.videoWidth / Math.max(1, video.videoHeight);
    if (aspectRatio < 1.30) return false;
    if (sourceMode === "video") return true;
    return getDisplayedPersonCount() >= 8;
}

function getNextPoseTile() {
    const tile = nextPoseTile;
    nextPoseTile = tile === "left" ? "right" : "left";
    return tile === "left"
        ? { mode: "tile", tile, xStart: 0, xScale: 0.58 }
        : { mode: "tile", tile, xStart: 0.42, xScale: 0.58 };
}

function transformPosePeople(people, meta) {
    const transformX = (value) => (meta.xStart + (value / YOLO_SIZE) * meta.xScale) * YOLO_SIZE;
    return people.map((person) => ({
        ...person,
        x1: transformX(person.x1),
        x2: transformX(person.x2),
        keypoints: person.keypoints.map((point) => ({
            ...point,
            x: transformX(point.x)
        }))
    }));
}

function dedupePosePeople(people) {
    const selected = [];
    const sorted = [...people].sort((a, b) => b.confidence - a.confidence);
    for (const person of sorted) {
        const duplicateIndex = selected.findIndex((candidate) => boxIou(candidate, person) > 0.48);
        if (duplicateIndex < 0) {
            selected.push(person);
            continue;
        }
        const existing = selected[duplicateIndex];
        const existingFacePoints = existing.keypoints.slice(0, 5)
            .filter((point) => point.confidence >= POSE_KEYPOINT_CONFIDENCE * 0.8).length;
        const newFacePoints = person.keypoints.slice(0, 5)
            .filter((point) => point.confidence >= POSE_KEYPOINT_CONFIDENCE * 0.8).length;
        if (newFacePoints > existingFacePoints) selected[duplicateIndex] = person;
    }
    return selected;
}

function estimatePoseHeadDirection(keypoints) {
    const valid = (index, threshold = POSE_KEYPOINT_CONFIDENCE) => {
        const point = keypoints[index];
        return point && Number.isFinite(point.x) && Number.isFinite(point.y)
            && point.confidence >= threshold;
    };
    const facePointCount = [0, 1, 2, 3, 4]
        .filter((index) => valid(index, POSE_KEYPOINT_CONFIDENCE * 0.8)).length;
    const nose = keypoints[0];

    if (valid(0) && facePointCount >= 3) {
        const referencePairs = [
            { left: 3, right: 4, threshold: POSE_HEAD_TURN_THRESHOLD },
            { left: 1, right: 2, threshold: 0.45 }
        ];

        for (const pair of referencePairs) {
            if (!valid(pair.left) || !valid(pair.right)) continue;
            const left = keypoints[pair.left];
            const right = keypoints[pair.right];
            const span = Math.abs(left.x - right.x);
            if (span < 4) continue;

            const centerX = (left.x + right.x) / 2;
            const offset = (nose.x - centerX) / span;
            if (offset > pair.threshold) {
                return { direction: "RIGHT", headDetected: true };
            }
            if (offset < -pair.threshold) {
                return { direction: "LEFT", headDetected: true };
            }
            return { direction: "CENTER", headDetected: true };
        }
    }

    return {
        direction: "UNKNOWN",
        headDetected: false
    };
}

function resetBehaviorTracking() {
    personTracks = new Map();
    documentTracks = new Map();
    nextPersonTrackId = 1;
    nextDocumentTrackId = 1;
    phoneTileResults.clear();
    nextPhoneTileIndex = 0;
}

function updateBehaviorTracking({ updateDirection, updateObjects }) {
    if (!sessionActive) return;

    const now = performance.now();
    const observations = buildPersonObservations();
    reconcilePersonTracks(observations, now);

    if (updateDirection) {
        for (const observation of observations) {
            const track = personTracks.get(observation.trackId);
            if (track) applyDirectionEvidence(track, observation.direction, now);
        }
    }

    if (updateObjects) {
        updatePhoneEvidence(observations, now);
        updateDocumentEvidence(observations, now);
    }

    cleanupBehaviorTracks(now);
}

function buildPersonObservations() {
    const yoloPeople = currentBoundingBoxes.filter((box) => box.classId === 0);
    const basePeople = yoloPeople.length ? yoloPeople : currentPosePeople;
    const usedFaces = new Set();
    const usedPoses = new Set();
    const scaleX = overlayCanvas.width / YOLO_SIZE || 1;
    const scaleY = overlayCanvas.height / YOLO_SIZE || 1;

    return basePeople.map((box) => {
        const poseIndex = yoloPeople.length ? findPoseForPerson(box, usedPoses) : -1;
        const pose = poseIndex >= 0 ? currentPosePeople[poseIndex] : (box.keypoints ? box : null);
        if (poseIndex >= 0) usedPoses.add(poseIndex);

        const faceIndex = findFaceInsidePerson(box, scaleX, scaleY, usedFaces);
        const face = faceIndex >= 0 ? currentFaceOrientations[faceIndex] : null;
        if (faceIndex >= 0) usedFaces.add(faceIndex);

        return {
            box,
            pose,
            face,
            direction: face ? face.direction : (pose ? pose.direction : "UNKNOWN"),
            trackId: null
        };
    });
}

function reconcilePersonTracks(observations, now) {
    const availableTracks = [...personTracks.values()]
        .filter((track) => now - track.lastSeenAt <= PERSON_TRACK_TTL_MS);
    const pairs = [];

    observations.forEach((observation, observationIndex) => {
        availableTracks.forEach((track) => {
            const overlap = boxIou(observation.box, track.box);
            const coverage = boxIntersectionOverMinArea(observation.box, track.box);
            const distance = boxCenterDistance(observation.box, track.box);
            const referenceSize = Math.max(
                72,
                Math.hypot(track.box.x2 - track.box.x1, track.box.y2 - track.box.y1) * 0.68
            );
            if (overlap < 0.03 && coverage < 0.18 && distance > referenceSize) return;
            pairs.push({
                observationIndex,
                trackId: track.id,
                score: overlap * 2.5 + coverage * 1.8
                    + Math.max(0, 1 - distance / Math.max(referenceSize, 1))
            });
        });
    });

    pairs.sort((a, b) => b.score - a.score);
    const assignedObservations = new Set();
    const assignedTracks = new Set();

    for (const pair of pairs) {
        if (assignedObservations.has(pair.observationIndex) || assignedTracks.has(pair.trackId)) continue;
        const observation = observations[pair.observationIndex];
        const track = personTracks.get(pair.trackId);
        if (!track) continue;
        assignObservationToTrack(observation, track, now);
        assignedObservations.add(pair.observationIndex);
        assignedTracks.add(pair.trackId);
    }

    observations.forEach((observation, index) => {
        if (assignedObservations.has(index)) return;
        const track = {
            id: nextPersonTrackId++,
            box: { ...observation.box },
            lastSeenAt: now,
            awayDirection: null,
            awaySince: 0,
            lastDirectionSeenAt: 0,
            centerSince: 0,
            awayAlerted: false,
            directionCandidate: null,
            directionCandidateCount: 0,
            directionCandidateSince: 0,
            directionCandidateLastSeenAt: 0,
            phoneSince: 0,
            phoneLastSeenAt: 0,
            phoneAlerted: false
        };
        personTracks.set(track.id, track);
        assignObservationToTrack(observation, track, now);
    });
}

function assignObservationToTrack(observation, track, now) {
    if (track.box) {
        const oldBox = track.box;
        const alpha = 0.72;
        track.box = {
            x1: oldBox.x1 * (1 - alpha) + observation.box.x1 * alpha,
            y1: oldBox.y1 * (1 - alpha) + observation.box.y1 * alpha,
            x2: oldBox.x2 * (1 - alpha) + observation.box.x2 * alpha,
            y2: oldBox.y2 * (1 - alpha) + observation.box.y2 * alpha
        };
    } else {
        track.box = { ...observation.box };
    }
    track.lastSeenAt = now;
    observation.trackId = track.id;
    observation.box.trackId = track.id;
    if (observation.pose) observation.pose.trackId = track.id;
}

function applyDirectionEvidence(track, direction, now) {
    const lookingAway = direction === "LEFT" || direction === "RIGHT";

    if (lookingAway) {
        const sameCandidate = track.directionCandidate === direction
            && now - track.directionCandidateLastSeenAt <= HEAD_DIRECTION_GRACE_MS;
        if (sameCandidate) {
            track.directionCandidateCount += 1;
        } else {
            track.directionCandidate = direction;
            track.directionCandidateCount = 1;
            track.directionCandidateSince = now;
        }
        track.directionCandidateLastSeenAt = now;
        if (track.directionCandidateCount < 3) return;

        const continues = track.awayDirection === direction
            && now - track.lastDirectionSeenAt <= HEAD_DIRECTION_GRACE_MS;
        if (!continues) {
            track.awayDirection = direction;
            track.awaySince = track.directionCandidateSince || now;
            track.awayAlerted = false;
        }
        track.lastDirectionSeenAt = now;
        track.centerSince = 0;

        const duration = now - track.awaySince;
        if (duration >= HEAD_TURN_ALERT_MS && !track.awayAlerted) {
            track.awayAlerted = true;
            const directionText = direction === "LEFT" ? "trái" : "phải";
            registerViolation(
                `HEAD_TURN_${track.id}`,
                `Người ${track.id} quay đầu quá 2 giây`,
                `Duy trì hướng ${directionText} ${(duration / 1000).toFixed(1)} giây`
            );
        }
        return;
    }

    if (direction === "CENTER") {
        resetDirectionCandidate(track);
        if (!track.centerSince) track.centerSince = now;
        if (now - track.centerSince >= HEAD_CENTER_RESET_MS) resetTrackDirection(track);
        return;
    }

    if (track.awayDirection && now - track.lastDirectionSeenAt > HEAD_DIRECTION_GRACE_MS) {
        resetTrackDirection(track);
    }
    if (track.directionCandidate
        && now - track.directionCandidateLastSeenAt > HEAD_DIRECTION_GRACE_MS) {
        resetDirectionCandidate(track);
    }
}

function resetTrackDirection(track) {
    track.awayDirection = null;
    track.awaySince = 0;
    track.lastDirectionSeenAt = 0;
    track.centerSince = 0;
    track.awayAlerted = false;
    resetDirectionCandidate(track);
}

function resetDirectionCandidate(track) {
    track.directionCandidate = null;
    track.directionCandidateCount = 0;
    track.directionCandidateSince = 0;
    track.directionCandidateLastSeenAt = 0;
}

function updatePhoneEvidence(observations, now) {
    const phones = getPhoneBoxes();
    const seenTrackIds = new Set();

    for (const phone of phones) {
        const ownerId = findObjectOwnerTrackId(phone, observations);
        phone.associatedTrackId = ownerId;
        if (!ownerId) continue;

        const track = personTracks.get(ownerId);
        if (!track) continue;
        const continues = track.phoneSince > 0 && now - track.phoneLastSeenAt <= PHONE_GRACE_MS;
        if (!continues) {
            track.phoneSince = now;
            track.phoneAlerted = false;
        }
        track.phoneLastSeenAt = now;
        seenTrackIds.add(ownerId);

        const duration = now - track.phoneSince;
        const requiredDuration = getDisplayedPersonCount() >= 3
            ? CLASSROOM_PHONE_ALERT_MS
            : PHONE_ALERT_MS;
        if (duration >= requiredDuration && !track.phoneAlerted) {
            track.phoneAlerted = true;
            registerViolation(
                `PHONE_USE_${ownerId}`,
                `Người ${ownerId} sử dụng điện thoại`,
                `Điện thoại được theo dõi liên tục ${(duration / 1000).toFixed(1)} giây`
            );
        }
    }

    for (const track of personTracks.values()) {
        if (seenTrackIds.has(track.id)) continue;
        if (track.phoneLastSeenAt && now - track.phoneLastSeenAt > PHONE_GRACE_MS) {
            track.phoneSince = 0;
            track.phoneLastSeenAt = 0;
            track.phoneAlerted = false;
        }
    }
}

function updateDocumentEvidence(observations, now) {
    const documents = currentBoundingBoxes.filter((box) => box.classId === 73);
    const activeTrackIds = new Set();

    for (const document of documents) {
        let bestTrack = null;
        let bestScore = -1;
        for (const track of documentTracks.values()) {
            if (activeTrackIds.has(track.id) || now - track.lastSeenAt > DOCUMENT_TRACK_TTL_MS) continue;
            const overlap = boxIou(document, track.box);
            const distance = boxCenterDistance(document, track.box);
            if (overlap < 0.04 && distance > 130) continue;
            const score = overlap * 3 + Math.max(0, 1 - distance / 130);
            if (score > bestScore) {
                bestScore = score;
                bestTrack = track;
            }
        }

        if (!bestTrack) {
            bestTrack = {
                id: nextDocumentTrackId++,
                box: { ...document },
                lastSeenAt: now,
                ownerId: null,
                ownerAnchorCenter: null,
                candidateOwnerId: null,
                candidateOwnerSince: 0,
                lastExchangeAt: 0
            };
            documentTracks.set(bestTrack.id, bestTrack);
        }

        bestTrack.box = { ...document };
        bestTrack.lastSeenAt = now;
        activeTrackIds.add(bestTrack.id);
        document.documentTrackId = bestTrack.id;

        const observedOwnerId = findObjectOwnerTrackId(document, observations);
        document.associatedTrackId = observedOwnerId;
        updateDocumentOwner(bestTrack, observedOwnerId, now);
    }
}

function updateDocumentOwner(documentTrack, observedOwnerId, now) {
    if (!observedOwnerId) {
        documentTrack.candidateOwnerId = null;
        documentTrack.candidateOwnerSince = 0;
        return;
    }

    if (observedOwnerId === documentTrack.ownerId) {
        documentTrack.candidateOwnerId = null;
        documentTrack.candidateOwnerSince = 0;
        return;
    }

    if (documentTrack.candidateOwnerId !== observedOwnerId) {
        documentTrack.candidateOwnerId = observedOwnerId;
        documentTrack.candidateOwnerSince = now;
        return;
    }

    if (now - documentTrack.candidateOwnerSince < DOCUMENT_OWNER_CONFIRM_MS) return;

    const previousOwnerId = documentTrack.ownerId;
    const currentCenter = boxCenter(documentTrack.box);

    if (previousOwnerId && documentTrack.ownerAnchorCenter) {
        const travel = Math.hypot(
            currentCenter.x - documentTrack.ownerAnchorCenter.x,
            currentCenter.y - documentTrack.ownerAnchorCenter.y
        );
        const documentWidth = Math.max(1, documentTrack.box.x2 - documentTrack.box.x1);
        if (travel < Math.max(55, documentWidth * 0.8)) {
            documentTrack.candidateOwnerId = null;
            documentTrack.candidateOwnerSince = 0;
            return;
        }
    }

    documentTrack.ownerId = observedOwnerId;
    documentTrack.ownerAnchorCenter = currentCenter;
    documentTrack.candidateOwnerId = null;
    documentTrack.candidateOwnerSince = 0;

    if (previousOwnerId && previousOwnerId !== observedOwnerId
        && now - documentTrack.lastExchangeAt > VIOLATION_COOLDOWN_MS) {
        documentTrack.lastExchangeAt = now;
        registerViolation(
            `DOCUMENT_EXCHANGE_${documentTrack.id}`,
            "Nghi trao đổi tài liệu",
            `Tài liệu chuyển từ người ${previousOwnerId} sang người ${observedOwnerId}`
        );
    }
}

function findObjectOwnerTrackId(objectBox, observations) {
    const centerX = (objectBox.x1 + objectBox.x2) / 2;
    const centerY = (objectBox.y1 + objectBox.y2) / 2;
    let bestTrackId = null;
    let bestScore = Number.POSITIVE_INFINITY;

    for (const observation of observations) {
        if (!observation.trackId) continue;
        const person = observation.box;
        const width = person.x2 - person.x1;
        const height = person.y2 - person.y1;
        const insideExpanded = centerX >= person.x1 - width * 0.18
            && centerX <= person.x2 + width * 0.18
            && centerY >= person.y1 - height * 0.08
            && centerY <= person.y2 + height * 0.20;
        const personCenterX = (person.x1 + person.x2) / 2;
        const personCenterY = (person.y1 + person.y2) / 2;
        const normalizedDistance = Math.hypot(centerX - personCenterX, centerY - personCenterY)
            / Math.max(1, Math.hypot(width, height));
        if (!insideExpanded && normalizedDistance > 0.72) continue;
        const score = normalizedDistance + (insideExpanded ? 0 : 1);
        if (score < bestScore) {
            bestScore = score;
            bestTrackId = observation.trackId;
        }
    }

    return bestTrackId;
}

function cleanupBehaviorTracks(now) {
    for (const [id, track] of personTracks) {
        if (now - track.lastSeenAt > PERSON_TRACK_TTL_MS) personTracks.delete(id);
    }
    for (const [id, track] of documentTracks) {
        if (now - track.lastSeenAt > DOCUMENT_TRACK_TTL_MS) documentTracks.delete(id);
    }
}

function boxIou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1));
    const areaB = Math.max(1, (b.x2 - b.x1) * (b.y2 - b.y1));
    return intersection / Math.max(1, areaA + areaB - intersection);
}

function boxIntersectionOverMinArea(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1));
    const areaB = Math.max(1, (b.x2 - b.x1) * (b.y2 - b.y1));
    return intersection / Math.max(1, Math.min(areaA, areaB));
}

function boxCenterDistance(a, b) {
    return Math.hypot(
        (a.x1 + a.x2 - b.x1 - b.x2) / 2,
        (a.y1 + a.y2 - b.y1 - b.y2) / 2
    );
}

function boxCenter(box) {
    return {
        x: (box.x1 + box.x2) / 2,
        y: (box.y1 + box.y2) / 2
    };
}

function drawOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawBoundingBoxes();
    drawTelemetryStrip();
}

function drawBoundingBoxes() {
    if (!currentBoundingBoxes.length && !currentPosePeople.length) return;

    const scaleX = overlayCanvas.width / YOLO_SIZE;
    const scaleY = overlayCanvas.height / YOLO_SIZE;
    const yoloPeople = currentBoundingBoxes.filter((box) => box.classId === 0);
    const people = [...(yoloPeople.length ? yoloPeople : currentPosePeople)]
        .sort((a, b) => a.x1 - b.x1);
    const objects = [
        ...currentBoundingBoxes.filter((box) => box.classId !== 0 && box.classId !== 67),
        ...getPhoneBoxes()
    ];
    const usedFaces = new Set();
    const usedPoses = new Set();

    people.forEach((box, index) => {
        const faceIndex = findFaceInsidePerson(box, scaleX, scaleY, usedFaces);
        const face = faceIndex >= 0 ? currentFaceOrientations[faceIndex] : null;
        const poseIndex = findPoseForPerson(box, usedPoses);
        const pose = poseIndex >= 0 ? currentPosePeople[poseIndex] : null;
        if (faceIndex >= 0) usedFaces.add(faceIndex);
        if (poseIndex >= 0) usedPoses.add(poseIndex);

        const trackId = box.trackId || (pose && pose.trackId) || index + 1;
        drawPersonBox(box, trackId, face, pose, scaleX, scaleY, personTracks.get(trackId));
    });

    for (const box of objects) {
        drawObjectBox(box, scaleX, scaleY);
    }
}

function findPoseForPerson(personBox, usedPoses) {
    let bestIndex = -1;
    let bestScore = 0;

    currentPosePeople.forEach((pose, index) => {
        if (usedPoses.has(index)) return;

        const intersectionX1 = Math.max(personBox.x1, pose.x1);
        const intersectionY1 = Math.max(personBox.y1, pose.y1);
        const intersectionX2 = Math.min(personBox.x2, pose.x2);
        const intersectionY2 = Math.min(personBox.y2, pose.y2);
        const intersection = Math.max(0, intersectionX2 - intersectionX1)
            * Math.max(0, intersectionY2 - intersectionY1);
        const areaA = Math.max(1, (personBox.x2 - personBox.x1) * (personBox.y2 - personBox.y1));
        const areaB = Math.max(1, (pose.x2 - pose.x1) * (pose.y2 - pose.y1));
        const union = areaA + areaB - intersection;
        const iou = union > 0 ? intersection / union : 0;
        const poseCenterX = (pose.x1 + pose.x2) / 2;
        const poseCenterY = (pose.y1 + pose.y2) / 2;
        const centerInside = poseCenterX >= personBox.x1 && poseCenterX <= personBox.x2
            && poseCenterY >= personBox.y1 && poseCenterY <= personBox.y2;
        const score = iou + (centerInside ? 1 : 0);

        if (score > bestScore) {
            bestScore = score;
            bestIndex = index;
        }
    });

    return bestScore >= 0.08 ? bestIndex : -1;
}

function findFaceInsidePerson(box, scaleX, scaleY, usedFaces) {
    const x1 = box.x1 * scaleX;
    const y1 = box.y1 * scaleY;
    const x2 = box.x2 * scaleX;
    const y2 = box.y2 * scaleY;
    const boxCenterX = (x1 + x2) / 2;
    const boxCenterY = (y1 + y2) / 2;

    let bestIndex = -1;
    let bestDistance = Number.POSITIVE_INFINITY;

    currentFaceOrientations.forEach((face, index) => {
        if (usedFaces.has(index)) return;

        const faceX = face.centerX * overlayCanvas.width;
        const faceY = face.centerY * overlayCanvas.height;
        if (faceX < x1 || faceX > x2 || faceY < y1 || faceY > y2) return;

        const distance = Math.hypot(faceX - boxCenterX, faceY - boxCenterY);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
        }
    });

    return bestIndex;
}

function drawPersonBox(box, personNumber, face, pose, scaleX, scaleY, track) {
    const x = box.x1 * scaleX;
    const y = box.y1 * scaleY;
    const width = (box.x2 - box.x1) * scaleX;
    const height = (box.y2 - box.y1) * scaleY;
    const rawDirection = face
        ? face.direction
        : (pose ? pose.direction : (box.direction || "UNKNOWN"));
    const now = performance.now();
    const trackedAway = track && track.awayDirection
        && now - track.lastDirectionSeenAt <= HEAD_DIRECTION_GRACE_MS;
    const direction = trackedAway ? track.awayDirection : rawDirection;
    const awayDuration = trackedAway ? Math.max(0, now - track.awaySince) : 0;
    const confirmedAway = awayDuration >= HEAD_TURN_ALERT_MS;
    const isLookingAway = direction === "LEFT" || direction === "RIGHT";
    const color = direction === "UNKNOWN"
        ? "#94a3b8"
        : confirmedAway ? "#ef4444" : isLookingAway ? "#f59e0b" : "#22c55e";

    overlayCtx.save();
    overlayCtx.lineWidth = direction === "UNKNOWN"
        ? Math.max(1, overlayCanvas.width / 700)
        : Math.max(2, overlayCanvas.width / 320);
    overlayCtx.strokeStyle = color;
    overlayCtx.fillStyle = direction === "UNKNOWN"
        ? "rgba(148, 163, 184, 0.015)"
        : isLookingAway ? "rgba(245, 158, 11, 0.08)" : "rgba(34, 197, 94, 0.05)";
    overlayCtx.strokeRect(x, y, width, height);
    overlayCtx.fillRect(x, y, width, height);

    const yawText = face ? ` ${Math.abs(face.yaw).toFixed(0)}°` : "";
    const crowded = getDisplayedPersonCount() > 8;
    const shortDirection = {
        LEFT: "TRÁI",
        RIGHT: "PHẢI",
        CENTER: "THẲNG",
        UNKNOWN: "CHƯA RÕ"
    }[direction];
    const durationText = isLookingAway ? ` ${(awayDuration / 1000).toFixed(1)}s` : "";
    const alertText = confirmedAway ? " ⚠" : "";
    const label = crowded
        ? (isLookingAway ? `N${personNumber} | ${shortDirection}${durationText}${alertText}` : "")
        : `NGƯỜI ${personNumber} | ${directionLabels[direction]}${yawText}${durationText}${alertText}`;
    if (label) drawOverlayLabel(x, y, label, color);
    if (pose && pose.headDetected && (!crowded || isLookingAway)) {
        drawPoseHead(pose, scaleX, scaleY, color);
    }
    overlayCtx.restore();
}

function drawPoseHead(person, scaleX, scaleY, color) {
    const facePoints = person.keypoints
        .slice(0, 5)
        .filter((point) => point.confidence >= POSE_KEYPOINT_CONFIDENCE * 0.8);
    if (!facePoints.length) return;

    const xs = facePoints.map((point) => point.x * scaleX);
    const ys = facePoints.map((point) => point.y * scaleY);
    const padding = Math.max(4, overlayCanvas.width / 180);
    const minX = Math.min(...xs) - padding;
    const maxX = Math.max(...xs) + padding;
    const minY = Math.min(...ys) - padding;
    const maxY = Math.max(...ys) + padding;

    overlayCtx.save();
    overlayCtx.strokeStyle = person.headDetected ? "#22d3ee" : color;
    overlayCtx.lineWidth = Math.max(1.5, overlayCanvas.width / 600);
    overlayCtx.strokeRect(minX, minY, Math.max(8, maxX - minX), Math.max(8, maxY - minY));

    for (const point of facePoints) {
        overlayCtx.beginPath();
        overlayCtx.arc(point.x * scaleX, point.y * scaleY, 2.2, 0, Math.PI * 2);
        overlayCtx.fillStyle = "#22d3ee";
        overlayCtx.fill();
    }
    overlayCtx.restore();
}

function drawObjectBox(box, scaleX, scaleY) {
    const x = box.x1 * scaleX;
    const y = box.y1 * scaleY;
    const width = (box.x2 - box.x1) * scaleX;
    const height = (box.y2 - box.y1) * scaleY;
    const isPhone = box.classId === 67;
    const isDocument = box.classId === 73;
    const color = isPhone ? "#ef4444" : isDocument ? "#0ea5e9" : "#8b5cf6";
    const objectName = isPhone
        ? "ĐIỆN THOẠI"
        : isDocument ? "TÀI LIỆU" : "LAPTOP / MÀN HÌNH";
    const ownerText = isPhone && box.associatedTrackId ? ` · N${box.associatedTrackId}` : "";

    overlayCtx.save();
    overlayCtx.lineWidth = Math.max(2, overlayCanvas.width / 320);
    overlayCtx.strokeStyle = color;
    overlayCtx.fillStyle = isPhone
        ? "rgba(239, 68, 68, 0.12)"
        : "rgba(139, 92, 246, 0.10)";
    overlayCtx.strokeRect(x, y, width, height);
    overlayCtx.fillRect(x, y, width, height);
    drawOverlayLabel(
        x,
        y,
        `${objectName} ${(box.confidence * 100).toFixed(0)}%${ownerText}`,
        color
    );
    overlayCtx.restore();
}

function drawOverlayLabel(x, y, text, backgroundColor) {
    const crowded = getDisplayedPersonCount() > 8;
    const fontSize = crowded
        ? Math.max(9, Math.min(12, overlayCanvas.width / 85))
        : Math.max(12, Math.min(18, overlayCanvas.width / 45));
    overlayCtx.font = `800 ${fontSize}px system-ui, sans-serif`;
    const paddingX = crowded ? 5 : 9;
    const labelHeight = fontSize + (crowded ? 7 : 12);
    const labelWidth = Math.min(
        overlayCanvas.width - 8,
        overlayCtx.measureText(text).width + paddingX * 2
    );
    const labelX = Math.max(4, Math.min(x, overlayCanvas.width - labelWidth - 4));
    const labelY = Math.max(4, y - labelHeight);

    overlayCtx.fillStyle = backgroundColor;
    overlayCtx.fillRect(labelX, labelY, labelWidth, labelHeight);
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(text, labelX + paddingX, labelY + fontSize + (crowded ? 2 : 4));
}

function getDisplayedPersonCount() {
    const yoloPeopleCount = currentBoundingBoxes.filter((box) => box.classId === 0).length;
    return yoloPeopleCount || currentPosePeople.length;
}

function getPhoneBoxes() {
    const crowded = getDisplayedPersonCount() >= 3;
    const candidates = [
        ...currentPhoneBoxes,
        ...currentBoundingBoxes.filter((box) => box.classId === 67)
    ].filter((box) => {
        if (!crowded) return true;
        const width = Math.max(0, box.x2 - box.x1);
        const height = Math.max(0, box.y2 - box.y1);
        const areaRatio = (width * height) / (YOLO_SIZE * YOLO_SIZE);
        const isPhoneEnsemble = String(box.detector || "").startsWith("phone-");
        const isPhoneTile = String(box.detector || "").endsWith("-tile");
        const confidenceFloor = isPhoneEnsemble
            ? (isPhoneTile ? 0.28 : 0.32)
            : 0.35;
        return box.confidence >= confidenceFloor && areaRatio <= 0.015;
    }).sort((a, b) => {
        const aDedicated = String(a.detector || "").startsWith("phone-");
        const bDedicated = String(b.detector || "").startsWith("phone-");
        if (aDedicated && !bDedicated) return -1;
        if (bDedicated && !aDedicated) return 1;
        return b.confidence - a.confidence;
    });
    const selected = [];
    for (const candidate of candidates) {
        if (selected.some((box) => boxIou(box, candidate) > 0.40)) continue;
        selected.push(candidate);
    }
    return selected;
}

function drawTelemetryStrip() {
    if (!overlayCanvas.width || !overlayCanvas.height) return;

    overlayCtx.save();

    const stripHeight = 36;

    overlayCtx.fillStyle = "rgba(2, 6, 23, 0.72)";
    overlayCtx.fillRect(
        0,
        overlayCanvas.height - stripHeight,
        overlayCanvas.width,
        stripHeight
    );

    overlayCtx.font = "bold 13px system-ui, sans-serif";
    const peopleCount = getDisplayedPersonCount();
    const now = performance.now();
    const phoneCount = [...personTracks.values()].filter(
        (track) => track.phoneSince && now - track.phoneLastSeenAt <= PHONE_GRACE_MS
            && now - track.phoneSince >= (
                getDisplayedPersonCount() >= 3 ? CLASSROOM_PHONE_ALERT_MS : PHONE_ALERT_MS
            )
    ).length;
    const awayCount = [...personTracks.values()].filter(
        (track) => track.awayDirection && now - track.lastDirectionSeenAt <= HEAD_DIRECTION_GRACE_MS
            && now - track.awaySince >= HEAD_TURN_ALERT_MS
    ).length;
    const documentCount = currentBoundingBoxes.filter((box) => box.classId === 73).length;

    overlayCtx.fillStyle = awayCount || phoneCount ? "#fbbf24" : "#86efac";
    const text = `NGƯỜI: ${peopleCount}  |  QUAY ĐẦU >2S: ${awayCount}  |  ĐIỆN THOẠI: ${phoneCount}  |  TÀI LIỆU: ${documentCount}`;

    overlayCtx.fillText(text, 14, overlayCanvas.height - 13);

    overlayCtx.restore();
}

function registerViolation(type, title, detail) {
    const now = Date.now();
    const last = lastViolationTimes[type] || 0;

    if (now - last < VIOLATION_COOLDOWN_MS) {
        return;
    }

    lastViolationTimes[type] = now;
    incidentTotal += 1;
    if (incidentCount) incidentCount.textContent = String(incidentTotal);

    if (feedStatus) {
        feedStatus.textContent = `${incidentTotal} sự kiện`;
        feedStatus.classList.add("error");
    }

    if (emptyFeed) {
        emptyFeed.classList.add("hidden");
    }

    const snapshot = createSnapshot();

    const card = document.createElement("article");
    card.className = "incident-card";

    const timeString = new Date(now).toLocaleTimeString("vi-VN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });

    card.innerHTML = `
        <div class="incident-inner">
            <img src="${snapshot}" alt="Ảnh minh chứng vi phạm" />
            <div>
                <h3 class="incident-title">${escapeHtml(title)}</h3>
                <p class="incident-detail">${escapeHtml(detail || "")}</p>
                <div class="incident-tag">${escapeHtml(type)}</div>
                <p class="incident-time">${timeString}</p>
                <div class="incident-actions">
                    <a href="${snapshot}" download="${timeString.replaceAll(":", "-")}-${type}.jpg">Tải ảnh</a>
                </div>
            </div>
        </div>
    `;
    card.dataset.snapshot = snapshot;
    card.dataset.violationType = type;

    logFeed.prepend(card);

    const cards = logFeed.querySelectorAll(".incident-card");

    if (cards.length > MAX_INCIDENT_CARDS) {
        cards[cards.length - 1].remove();
    }
}

function createSnapshot() {
    const width = video.videoWidth || rawCanvas.width || VIDEO_TARGET_WIDTH;
    const height = video.videoHeight || rawCanvas.height || VIDEO_TARGET_HEIGHT;

    snapshotCanvas.width = width;
    snapshotCanvas.height = height;

    snapshotCtx.clearRect(0, 0, width, height);

    try {
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            snapshotCtx.drawImage(sourceCanvas, 0, 0, width, height);

            if (overlayCanvas.width && overlayCanvas.height) {
                snapshotCtx.drawImage(overlayCanvas, 0, 0, width, height);
            }
        } else {
            snapshotCtx.fillStyle = "#020617";
            snapshotCtx.fillRect(0, 0, width, height);
            snapshotCtx.fillStyle = "#e5e7eb";
            snapshotCtx.font = "bold 24px system-ui, sans-serif";
            snapshotCtx.fillText("No active video frame", 28, 56);
        }
    } catch (error) {
        snapshotCtx.fillStyle = "#020617";
        snapshotCtx.fillRect(0, 0, width, height);
        snapshotCtx.fillStyle = "#e5e7eb";
        snapshotCtx.font = "bold 24px system-ui, sans-serif";
        snapshotCtx.fillText("Snapshot unavailable", 28, 56);
    }

    return snapshotCanvas.toDataURL("image/jpeg", 0.65);
}

function purgeLogs() {
    lastViolationTimes = {};
    incidentTotal = 0;
    if (incidentCount) incidentCount.textContent = "0";

    const cards = logFeed.querySelectorAll(".incident-card");

    for (const card of cards) {
        card.remove();
    }

    if (emptyFeed) {
        emptyFeed.classList.remove("hidden");
    }

    if (feedStatus) {
        feedStatus.textContent = "An toàn";
        feedStatus.classList.remove("error");
    }
}

function exportViolationsAsZip() {
    const cards = Array.from(logFeed.querySelectorAll(".incident-card"));

    if (!cards.length) {
        if (feedStatus) feedStatus.textContent = "Chưa có minh chứng";
        return;
    }

    const files = cards.map((card, index) => {
        const snapshot = card.dataset.snapshot;
        if (!snapshot) return null;

        const type = (card.dataset.violationType || `violation-${index + 1}`).replace(/\s+/g, "_").toLowerCase();
        const timeLabel = card.querySelector(".incident-time")?.textContent.replace(/[: ]/g, "_") || `event_${index + 1}`;
        const filename = `violation_${index + 1}_${type}_${timeLabel}.jpg`;

        return {
            name: filename,
            data: base64ToUint8Array(snapshot)
        };
    }).filter(Boolean);

    if (!files.length) {
        if (feedStatus) feedStatus.textContent = "Không có ảnh để xuất";
        return;
    }

    const zipBlob = createZipBlob(files);
    const downloadName = `violations-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.zip`;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(zipBlob);
    link.download = downloadName;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function base64ToUint8Array(dataUrl) {
    const parts = dataUrl.split(",");
    const base64 = parts[1] || "";
    const binary = atob(base64);
    const buffer = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i += 1) {
        buffer[i] = binary.charCodeAt(i);
    }

    return buffer;
}

function crc32(bytes) {
    let crc = -1;

    for (let i = 0; i < bytes.length; i += 1) {
        crc ^= bytes[i];
        for (let j = 0; j < 8; j += 1) {
            crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
        }
    }

    return (crc ^ -1) >>> 0;
}

function createZipBlob(files) {
    const encoder = new TextEncoder();
    const localEntries = [];
    const centralEntries = [];
    let offset = 0;

    for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const crc = crc32(file.data);
        const header = new Uint8Array(30 + nameBytes.length);
        const view = new DataView(header.buffer);

        view.setUint32(0, 0x04034b50, true);
        view.setUint16(4, 20, true);
        view.setUint16(6, 0, true);
        view.setUint16(8, 0, true);
        view.setUint16(10, 0, true);
        view.setUint32(14, crc, true);
        view.setUint32(18, file.data.length, true);
        view.setUint32(22, file.data.length, true);
        view.setUint16(26, nameBytes.length, true);
        header.set(nameBytes, 30);

        localEntries.push(header, file.data);

        const central = new Uint8Array(46 + nameBytes.length);
        const centralView = new DataView(central.buffer);

        centralView.setUint32(0, 0x02014b50, true);
        centralView.setUint16(4, 20, true);
        centralView.setUint16(6, 20, true);
        centralView.setUint16(8, 0, true);
        centralView.setUint16(10, 0, true);
        centralView.setUint16(12, 0, true);
        centralView.setUint32(16, crc, true);
        centralView.setUint32(20, file.data.length, true);
        centralView.setUint32(24, file.data.length, true);
        centralView.setUint16(28, nameBytes.length, true);
        centralView.setUint32(42, offset, true);
        central.set(nameBytes, 46);

        centralEntries.push(central);
        offset += header.length + file.data.length;
    }

    const directorySize = centralEntries.reduce((sum, entry) => sum + entry.length, 0);
    const endRecord = new Uint8Array(22);
    const endView = new DataView(endRecord.buffer);

    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, directorySize, true);
    endView.setUint32(16, offset, true);

    return new Blob([...localEntries, ...centralEntries, endRecord], { type: "application/zip" });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
