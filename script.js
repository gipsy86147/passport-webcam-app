const limitsByMode = {
  photo: 250 * 1024,
  signature: 100 * 1024,
};

const resolutionByMode = {
  photo: { width: 600, height: 600 },
  signature: { width: 600, height: 250 },
};

const modeEl = document.getElementById("mode");
const startBtn = document.getElementById("startCamera");
const captureBtn = document.getElementById("capture");
const retakeBtn = document.getElementById("retake");
const downloadBtn = document.getElementById("download");
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

async function ensureBodyPixModel() {
  if (!window.bodyPix || !window.tf) {
    return null;
  }

  if (!bodyPixModelPromise) {
    bodyPixModelPromise = (async () => {
      try {
        await window.tf.setBackend("webgl");
        await window.tf.ready();
      } catch (_) {
        // Fallback to whichever backend is available.
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
    return { canvas: sourceCanvas, applied: false };
  }

  try {
    const segmentation = await model.segmentPerson(sourceCanvas, {
      flipHorizontal: false,
      internalResolution: "medium",
      segmentationThreshold: 0.7,
      maxDetections: 1,
      scoreThreshold: 0.3,
      nmsRadius: 20,
    });

    if (!segmentation?.data) {
      return { canvas: sourceCanvas, applied: false };
    }

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const srcCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    const src = srcCtx.getImageData(0, 0, width, height).data;

    const whiteCanvas = document.createElement("canvas");
    whiteCanvas.width = width;
    whiteCanvas.height = height;
    const outCtx = whiteCanvas.getContext("2d");
    const outImage = outCtx.createImageData(width, height);
    const out = outImage.data;

    for (let pixel = 0; pixel < segmentation.data.length; pixel += 1) {
      const i = pixel * 4;
      if (segmentation.data[pixel] === 1) {
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

    outCtx.putImageData(outImage, 0, 0);
    return { canvas: whiteCanvas, applied: true };
  } catch (_) {
    return { canvas: sourceCanvas, applied: false };
  }
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

async function detectPrimaryFace(sourceCanvas) {
  if (typeof window.FaceDetector !== "function") {
    return null;
  }

  try {
    const detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    const detectedFaces = await detector.detect(sourceCanvas);
    return detectedFaces.length > 0 ? detectedFaces[0].boundingBox : null;
  } catch (_) {
    return null;
  }
}

async function resolveFocusPoint(sourceCanvas, mode) {
  if (mode !== "photo") {
    return { x: 0.5, y: 0.5, faceCentered: false };
  }

  const face = await detectPrimaryFace(sourceCanvas);
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

async function compressToLimit(sourceCanvas, maxBytes, initialWidth, initialHeight, focusPoint) {
  const workCanvas = document.createElement("canvas");
  let width = initialWidth;
  let height = initialHeight;

  drawCover(sourceCanvas, workCanvas, width, height, focusPoint.x, focusPoint.y);

  for (let round = 0; round < 8; round += 1) {
    for (let quality = 0.92; quality >= 0.3; quality -= 0.08) {
      const blob = await canvasToJpegBlob(workCanvas, quality);
      if (blob && blob.size <= maxBytes) {
        return { blob, width, height, quality };
      }
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

async function captureImage() {
  if (!stream) {
    setStatus("Start camera first.");
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

  const mode = modeEl.value;
  const maxBytes = limitsByMode[mode];
  const targetRes = resolutionByMode[mode];
  const focusPoint = await resolveFocusPoint(captureCanvas, mode);
  const wantsWhiteBg = mode === "photo" && whiteBgEl.checked;
  let sourceForOutput = captureCanvas;
  let whiteBgApplied = false;

  if (wantsWhiteBg) {
    setStatus("Applying white background...", false);
    const whiteBgResult = await replaceBackgroundWithWhite(captureCanvas);
    sourceForOutput = whiteBgResult.canvas;
    whiteBgApplied = whiteBgResult.applied;
  }

  const result = await compressToLimit(
    sourceForOutput,
    maxBytes,
    targetRes.width,
    targetRes.height,
    focusPoint,
  );

  if (!result) {
    setStatus("Could not compress under limit. Move closer to plain background and retry.");
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
      ? " White background applied."
      : " White background could not be applied; original background kept."
    : "";
  setStatus(
    `Captured ${mode} as JPG in ${formatKB(lastCaptureBlob.size)} (limit ${formatKB(maxBytes)}).${centeringHint}${backgroundHint}`,
    true,
  );
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
retakeBtn.addEventListener("click", resetCapture);
downloadBtn.addEventListener("click", downloadCapture);
window.addEventListener("beforeunload", stopCamera);

updateModeLayout();
updateLimitText();
