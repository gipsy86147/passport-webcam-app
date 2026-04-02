const limitsByMode = {
  photo: 250 * 1024,
  signature: 100 * 1024,
};

const resolutionByMode = {
  photo: { width: 630, height: 810 },
  signature: { width: 600, height: 250 },
};

const modeEl = document.getElementById("mode");
const startBtn = document.getElementById("startCamera");
const captureBtn = document.getElementById("capture");
const uploadBtn = document.getElementById("uploadImage");
const retakeBtn = document.getElementById("retake");
const downloadBtn = document.getElementById("download");
const imageInputEl = document.getElementById("imageInput");
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const previewEl = document.getElementById("outputPreview");
const cameraFrameEl = document.getElementById("cameraFrame");
const outputFrameEl = document.getElementById("outputFrame");
const outputSizeEl = document.getElementById("outputSize");
const limitValueEl = document.getElementById("limitValue");
const whiteBgEl = document.getElementById("whiteBg");
const captureCanvas = document.getElementById("captureCanvas");

let stream = null;
let lastCaptureBlob = null;
let bodyPixModelPromise = null;
let faceDetectorInstance = null;

function formatKB(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function setStatus(message, ok = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("ok", ok);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function updateLimitText() {
  const mode = modeEl.value;
  limitValueEl.textContent = formatKB(limitsByMode[mode]);
}

function updateModeLayout() {
  const mode = modeEl.value;
  cameraFrameEl.dataset.mode = mode;
  outputFrameEl.dataset.mode = mode;
  whiteBgEl.disabled = mode !== "photo";
}

async function canvasToJpegBlob(canvas, quality) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

function getFaceDetector() {
  if (typeof window.FaceDetector !== "function") {
    return null;
  }

  if (!faceDetectorInstance) {
    try {
      faceDetectorInstance = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    } catch (_) {
      try {
        faceDetectorInstance = new FaceDetector();
      } catch (_) {
        faceDetectorInstance = null;
      }
    }
  }

  return faceDetectorInstance;
}

async function ensureBodyPixModel() {
  if (!window.bodyPix || !window.tf) {
    return null;
  }

  if (!bodyPixModelPromise) {
    bodyPixModelPromise = (async () => {
      let backendReady = false;
      try {
        await window.tf.setBackend("webgl");
        await window.tf.ready();
        backendReady = true;
      } catch (_) {
        // Fallback below.
      }

      if (!backendReady) {
        try {
          await window.tf.setBackend("cpu");
          await window.tf.ready();
          backendReady = true;
        } catch (_) {
          // Leave backend selection to tfjs defaults.
        }
      }

      return window.bodyPix.load({
        architecture: "MobileNetV1",
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2,
      });
    })();
  }

  try {
    return await bodyPixModelPromise;
  } catch (_) {
    return null;
  }
}

async function replaceBackgroundWithWhite(sourceCanvas) {
  const model = await ensureBodyPixModel();
  if (!model) {
    return applyHeuristicWhiteBackground(sourceCanvas);
  }

  try {
    const sourceWidth = sourceCanvas.width;
    const sourceHeight = sourceCanvas.height;
    const segWidth = Math.min(640, sourceWidth);
    const segHeight = Math.max(360, Math.round((segWidth * sourceHeight) / sourceWidth));
    const segCanvas = document.createElement("canvas");
    segCanvas.width = segWidth;
    segCanvas.height = segHeight;
    segCanvas.getContext("2d").drawImage(sourceCanvas, 0, 0, segWidth, segHeight);

    const segmentation = await model.segmentPerson(segCanvas, {
      flipHorizontal: false,
      internalResolution: "low",
      segmentationThreshold: 0.7,
      maxDetections: 1,
      scoreThreshold: 0.3,
      nmsRadius: 20,
    });

    if (!segmentation?.data) {
      return { canvas: sourceCanvas, applied: false };
    }

    const width = sourceWidth;
    const height = sourceHeight;
    const segData = segmentation.data;
    const srcCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    const src = srcCtx.getImageData(0, 0, width, height).data;

    const whiteCanvas = document.createElement("canvas");
    whiteCanvas.width = width;
    whiteCanvas.height = height;
    const outCtx = whiteCanvas.getContext("2d");
    const outImage = outCtx.createImageData(width, height);
    const out = outImage.data;

    for (let y = 0; y < height; y += 1) {
      const segY = Math.min(segHeight - 1, Math.floor((y * segHeight) / height));
      const rowOffset = y * width;
      const segRowOffset = segY * segWidth;

      for (let x = 0; x < width; x += 1) {
        const segX = Math.min(segWidth - 1, Math.floor((x * segWidth) / width));
        const segPixel = segData[segRowOffset + segX];
        const i = (rowOffset + x) * 4;

        if (segPixel === 1) {
          out[i] = src[i];
          out[i + 1] = src[i + 1];
          out[i + 2] = src[i + 2];
        } else {
          out[i] = 255;
          out[i + 1] = 255;
          out[i + 2] = 255;
        }
        out[i + 3] = 255;
      }
    }

    outCtx.putImageData(outImage, 0, 0);
    return { canvas: whiteCanvas, applied: true, method: "ai" };
  } catch (_) {
    return applyHeuristicWhiteBackground(sourceCanvas);
  }
}

function applyHeuristicWhiteBackground(sourceCanvas) {
  const width = sourceCanvas.width;
  const height = sourceCanvas.height;
  const ctx = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const srcImage = ctx.getImageData(0, 0, width, height);
  const src = srcImage.data;

  const edgeStep = Math.max(4, Math.floor(Math.min(width, height) / 150));
  const margin = Math.max(2, Math.floor(Math.min(width, height) * 0.03));
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let edgeCount = 0;

  for (let x = margin; x < width - margin; x += edgeStep) {
    const topI = ((margin * width) + x) * 4;
    const bottomI = (((height - margin - 1) * width) + x) * 4;
    rSum += src[topI] + src[bottomI];
    gSum += src[topI + 1] + src[bottomI + 1];
    bSum += src[topI + 2] + src[bottomI + 2];
    edgeCount += 2;
  }

  for (let y = margin; y < height - margin; y += edgeStep) {
    const leftI = ((y * width) + margin) * 4;
    const rightI = ((y * width) + (width - margin - 1)) * 4;
    rSum += src[leftI] + src[rightI];
    gSum += src[leftI + 1] + src[rightI + 1];
    bSum += src[leftI + 2] + src[rightI + 2];
    edgeCount += 2;
  }

  if (edgeCount === 0) {
    return { canvas: sourceCanvas, applied: false, method: "none" };
  }

  const bgR = rSum / edgeCount;
  const bgG = gSum / edgeCount;
  const bgB = bSum / edgeCount;
  const bgLuma = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;

  const outCanvas = document.createElement("canvas");
  outCanvas.width = width;
  outCanvas.height = height;
  const outCtx = outCanvas.getContext("2d");
  const outImage = outCtx.createImageData(width, height);
  const out = outImage.data;

  let whitened = 0;
  const chromaTolerance = bgLuma > 170 ? 78 : 58;
  const brightLumaCutoff = Math.max(190, bgLuma - 8);
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i];
    const g = src[i + 1];
    const b = src[i + 2];
    const luma = 0.299 * r + 0.587 * g + 0.114 * b;
    const colorDiff = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);
    const nearBackground = colorDiff < chromaTolerance && luma > brightLumaCutoff;

    if (nearBackground) {
      out[i] = 255;
      out[i + 1] = 255;
      out[i + 2] = 255;
      whitened += 1;
    } else {
      out[i] = r;
      out[i + 1] = g;
      out[i + 2] = b;
    }
    out[i + 3] = 255;
  }

  outCtx.putImageData(outImage, 0, 0);
  const whitenedRatio = whitened / (width * height);
  if (whitenedRatio < 0.12) {
    return { canvas: sourceCanvas, applied: false, method: "none" };
  }

  return { canvas: outCanvas, applied: true, method: "heuristic" };
}

function drawCover(sourceCanvas, targetCanvas, width, height, focusX = 0.5, focusY = 0.5) {
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const srcRatio = srcW / srcH;
  const targetRatio = width / height;

  let cropW = srcW;
  let cropH = srcH;

  if (srcRatio > targetRatio) {
    cropW = srcH * targetRatio;
  } else {
    cropH = srcW / targetRatio;
  }

  const cropX = clamp(srcW * focusX - cropW / 2, 0, srcW - cropW);
  const cropY = clamp(srcH * focusY - cropH / 2, 0, srcH - cropH);

  targetCanvas.width = width;
  targetCanvas.height = height;
  const ctx = targetCanvas.getContext("2d");
  ctx.drawImage(sourceCanvas, cropX, cropY, cropW, cropH, 0, 0, width, height);
}

async function detectPrimaryFaceData(sourceCanvas) {
  const detector = getFaceDetector();
  if (!detector) {
    return null;
  }

  try {
    const detectedFaces = await detector.detect(sourceCanvas);
    return detectedFaces.length > 0 ? detectedFaces[0] : null;
  } catch (_) {
    return null;
  }
}

function getLandmark(landmarks, candidates) {
  if (!Array.isArray(landmarks)) {
    return null;
  }

  const normalizedCandidates = candidates.map((name) => name.toLowerCase());
  return (
    landmarks.find((point) => normalizedCandidates.includes(String(point.type || "").toLowerCase())) ||
    null
  );
}

function evaluateFrontalPose(faceData) {
  if (!faceData?.boundingBox || !Array.isArray(faceData.landmarks)) {
    return { ok: true, confident: false };
  }

  const leftEye = getLandmark(faceData.landmarks, ["leftEye", "eyeLeft", "eye"]);
  const rightEye = getLandmark(faceData.landmarks, ["rightEye", "eyeRight"]);
  const nose = getLandmark(faceData.landmarks, ["noseTip", "nose"]);

  if (!leftEye || !rightEye || !nose) {
    return { ok: true, confident: false };
  }

  const eyeDx = Math.max(1, Math.abs(leftEye.x - rightEye.x));
  const eyeRoll = Math.abs((leftEye.y - rightEye.y) / eyeDx);
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const noseOffsetRatio = Math.abs(nose.x - eyeMidX) / Math.max(1, faceData.boundingBox.width);
  const ok = eyeRoll <= 0.08 && noseOffsetRatio <= 0.1;

  return {
    ok,
    confident: true,
    eyeRoll,
    noseOffsetRatio,
  };
}

async function resolveFocusPoint(sourceCanvas, mode) {
  if (mode !== "photo") {
    return { x: 0.5, y: 0.5, faceCentered: false };
  }

  const faceData = await detectPrimaryFaceData(sourceCanvas);
  const face = faceData?.boundingBox;
  if (!face) {
    return { x: 0.5, y: 0.47, faceCentered: false };
  }

  const faceCenterX = (face.x + face.width / 2) / sourceCanvas.width;
  const faceCenterY = (face.y + face.height * 0.42) / sourceCanvas.height;

  return {
    x: clamp(faceCenterX, 0, 1),
    y: clamp(faceCenterY, 0, 1),
    faceCentered: true,
  };
}

async function normalizePhotoComposition(sourceCanvas, width, height, focusPoint) {
  const baseCanvas = createCanvas(width, height);
  drawCover(sourceCanvas, baseCanvas, width, height, focusPoint.x, focusPoint.y);

  const faceData = await detectPrimaryFaceData(baseCanvas);
  const bbox = faceData?.boundingBox;
  if (!bbox) {
    return {
      canvas: baseCanvas,
      notes: ["Face not detected clearly. Keep your full face visible."],
    };
  }

  const faceWidthRatio = bbox.width / width;
  const targetFaceRatio = 0.42;
  const maxFaceRatio = 0.5;
  const minFaceRatio = 0.3;
  const notes = [];
  let outputCanvas = baseCanvas;

  if (faceWidthRatio > maxFaceRatio) {
    const scale = clamp(targetFaceRatio / faceWidthRatio, 0.72, 0.98);
    const scaledW = Math.round(width * scale);
    const scaledH = Math.round(height * scale);
    const adjusted = createCanvas(width, height);
    const ctx = adjusted.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const faceCenterX = bbox.x + bbox.width / 2;
    const faceCenterY = bbox.y + bbox.height / 2;
    const offsetX = width / 2 - faceCenterX * scale;
    const offsetY = height * 0.43 - faceCenterY * scale;

    ctx.drawImage(baseCanvas, offsetX, offsetY, scaledW, scaledH);
    outputCanvas = adjusted;
    notes.push("Head size auto-adjusted for ICAO framing.");
  } else if (faceWidthRatio < minFaceRatio) {
    notes.push("Move slightly closer to the camera so face size is acceptable.");
  }

  const finalFaceData = await detectPrimaryFaceData(outputCanvas);
  const pose = evaluateFrontalPose(finalFaceData || faceData);
  if (pose.confident && !pose.ok) {
    notes.push("Face camera directly and keep your head straight.");
  }

  return {
    canvas: outputCanvas,
    notes,
  };
}

async function compressToLimit(
  sourceCanvas,
  maxBytes,
  initialWidth,
  initialHeight,
  focusPoint,
  allowResize = true,
) {
  const workCanvas = document.createElement("canvas");
  let width = initialWidth;
  let height = initialHeight;

  drawCover(sourceCanvas, workCanvas, width, height, focusPoint.x, focusPoint.y);

  for (let round = 0; round < 8; round += 1) {
    for (let quality = 0.92; quality >= 0.2; quality -= 0.06) {
      const blob = await canvasToJpegBlob(workCanvas, quality);
      if (blob && blob.size <= maxBytes) {
        return { blob, width, height, quality };
      }
    }

    if (!allowResize) {
      break;
    }

    width = Math.round(width * 0.9);
    height = Math.round(height * 0.9);
    if (width < 220 || height < 120) {
      break;
    }

    drawCover(sourceCanvas, workCanvas, width, height, focusPoint.x, focusPoint.y);
  }

  return null;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not support webcam access.");
    return;
  }

  try {
    stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });

    videoEl.srcObject = stream;
    captureBtn.disabled = false;
    setStatus("Camera started. Align and click Capture.", true);
  } catch (err) {
    setStatus(`Unable to access camera: ${err.message || err}`);
  }
}

function stopCamera() {
  if (!stream) {
    return;
  }

  stream.getTracks().forEach((track) => track.stop());
  stream = null;
}

async function processSourceCanvas(sourceCanvas, sourceLabel) {
  if (!sourceCanvas?.width || !sourceCanvas?.height) {
    setStatus("Selected image is not valid.");
    return;
  }

  const mode = modeEl.value;
  const maxBytes = limitsByMode[mode];
  const targetRes = resolutionByMode[mode];
  const shouldKeepExactDimensions = mode === "photo";
  const focusPoint = await resolveFocusPoint(sourceCanvas, mode);
  const wantsWhiteBg = mode === "photo" && whiteBgEl.checked;
  let sourceForOutput = sourceCanvas;
  let whiteBgApplied = false;
  let whiteBgMethod = "none";
  let complianceNotes = [];

  if (wantsWhiteBg) {
    setStatus("Applying white background...", false);
    const whiteBgResult = await replaceBackgroundWithWhite(sourceCanvas);
    sourceForOutput = whiteBgResult.canvas;
    whiteBgApplied = whiteBgResult.applied;
    whiteBgMethod = whiteBgResult.method || "none";
  }

  if (mode === "photo") {
    const normalized = await normalizePhotoComposition(
      sourceForOutput,
      targetRes.width,
      targetRes.height,
      focusPoint,
    );
    sourceForOutput = normalized.canvas;
    complianceNotes = normalized.notes;
  }

  const result = await compressToLimit(
    sourceForOutput,
    maxBytes,
    targetRes.width,
    targetRes.height,
    mode === "photo" ? { x: 0.5, y: 0.5 } : focusPoint,
    !shouldKeepExactDimensions,
  );

  if (!result) {
    const dimensionHint = shouldKeepExactDimensions
      ? `at ${targetRes.width}x${targetRes.height}`
      : "under current settings";
    setStatus(`Could not compress ${dimensionHint} under limit. Retry with plain background and better lighting.`);
    return;
  }

  const fileName = `${mode}-${Date.now()}.jpg`;
  lastCaptureBlob = new File([result.blob], fileName, { type: "image/jpeg" });

  if (previewEl.src) {
    URL.revokeObjectURL(previewEl.src);
  }
  previewEl.src = URL.createObjectURL(lastCaptureBlob);
  outputSizeEl.textContent = `${formatKB(lastCaptureBlob.size)} (${result.width}x${result.height})`;

  downloadBtn.disabled = false;
  retakeBtn.disabled = false;
  const centeringHint = focusPoint.faceCentered
    ? " Face auto-centered."
    : " Centered using frame guide.";
  const backgroundHint = wantsWhiteBg
    ? whiteBgApplied
      ? whiteBgMethod === "heuristic"
        ? " White background applied (fallback mode)."
        : " White background applied."
      : " White background could not be applied; original background kept."
    : "";
  const complianceHint = complianceNotes.length > 0 ? ` ${complianceNotes.join(" ")}` : "";
  setStatus(
    `${sourceLabel} ${mode} as JPG in ${formatKB(lastCaptureBlob.size)} (limit ${formatKB(maxBytes)}).${centeringHint}${backgroundHint}${complianceHint}`,
    true,
  );
}

async function captureImage() {
  if (!stream) {
    setStatus("Start camera first, or use Upload Image.");
    return;
  }

  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  if (!videoW || !videoH) {
    setStatus("Camera is not ready yet. Please try again.");
    return;
  }

  captureCanvas.width = videoW;
  captureCanvas.height = videoH;
  captureCanvas.getContext("2d").drawImage(videoEl, 0, 0, videoW, videoH);
  await processSourceCanvas(captureCanvas, "Captured");
}

async function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not read the selected image."));
    };
    img.src = objectUrl;
  });
}

async function handleImageUpload(event) {
  const file = event.target.files?.[0];
  imageInputEl.value = "";

  if (!file) {
    return;
  }

  if (!file.type.startsWith("image/")) {
    setStatus("Please upload a valid image file.");
    return;
  }

  try {
    setStatus("Preparing uploaded image...", false);
    const img = await loadImageFromFile(file);
    const uploadCanvas = createCanvas(img.naturalWidth || img.width, img.naturalHeight || img.height);
    uploadCanvas.getContext("2d").drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);
    await processSourceCanvas(uploadCanvas, "Converted");
  } catch (error) {
    setStatus(`Upload failed: ${error.message || error}`);
  }
}

function resetCapture() {
  if (previewEl.src) {
    URL.revokeObjectURL(previewEl.src);
  }

  previewEl.removeAttribute("src");
  lastCaptureBlob = null;
  outputSizeEl.textContent = "-";
  downloadBtn.disabled = true;
  retakeBtn.disabled = true;
  setStatus("Ready to capture again.", true);
}

function downloadCapture() {
  if (!lastCaptureBlob) {
    return;
  }

  const a = document.createElement("a");
  a.href = URL.createObjectURL(lastCaptureBlob);
  a.download = lastCaptureBlob.name;
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}

modeEl.addEventListener("change", () => {
  updateModeLayout();
  updateLimitText();
  resetCapture();
});

startBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureImage);
uploadBtn.addEventListener("click", () => imageInputEl.click());
imageInputEl.addEventListener("change", handleImageUpload);
retakeBtn.addEventListener("click", resetCapture);
downloadBtn.addEventListener("click", downloadCapture);
window.addEventListener("beforeunload", stopCamera);

updateModeLayout();
updateLimitText();
