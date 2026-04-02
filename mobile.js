const PHOTO_WIDTH = 630;
const PHOTO_HEIGHT = 810;
const PHOTO_MAX_BYTES = 250 * 1024;

const startBtn = document.getElementById("startBackCamera");
const captureBtn = document.getElementById("capturePhoto");
const retakeBtn = document.getElementById("retakePhoto");
const downloadBtn = document.getElementById("downloadPhoto");
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const previewEl = document.getElementById("outputPreview");
const outputSizeEl = document.getElementById("outputSize");
const captureCanvas = document.getElementById("captureCanvas");

let stream = null;
let lastCaptureBlob = null;
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

function getFaceDetector() {
  if (typeof window.FaceDetector !== "function") {
    return null;
  }

  if (!faceDetectorInstance) {
    try {
      faceDetectorInstance = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    } catch (_) {
      faceDetectorInstance = null;
    }
  }

  return faceDetectorInstance;
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

async function resolveFocusPoint(sourceCanvas) {
  const faceData = await detectPrimaryFaceData(sourceCanvas);
  const box = faceData?.boundingBox;
  if (!box) {
    return { x: 0.5, y: 0.47, faceCentered: false };
  }

  const faceCenterX = (box.x + box.width / 2) / sourceCanvas.width;
  const faceCenterY = (box.y + box.height * 0.42) / sourceCanvas.height;

  return {
    x: clamp(faceCenterX, 0, 1),
    y: clamp(faceCenterY, 0, 1),
    faceCentered: true,
  };
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

async function canvasToJpegBlob(canvas, quality) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

async function compressToPhotoLimit(sourceCanvas, focusPoint) {
  const workCanvas = document.createElement("canvas");
  drawCover(sourceCanvas, workCanvas, PHOTO_WIDTH, PHOTO_HEIGHT, focusPoint.x, focusPoint.y);

  for (let quality = 0.92; quality >= 0.18; quality -= 0.04) {
    const blob = await canvasToJpegBlob(workCanvas, quality);
    if (blob && blob.size <= PHOTO_MAX_BYTES) {
      return { blob, quality };
    }
  }

  return null;
}

async function startBackCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not support webcam access.");
    return;
  }

  const attempts = [
    {
      video: {
        facingMode: { exact: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    },
    {
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    },
    { video: true, audio: false },
  ];

  stopCamera();

  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = stream;
      captureBtn.disabled = false;
      setStatus("Back camera started. Align your face inside the outline and capture.", true);
      return;
    } catch (_) {
      // Try next constraints.
    }
  }

  setStatus("Unable to access back camera. Check permissions and retry.");
}

function stopCamera() {
  if (!stream) {
    return;
  }

  stream.getTracks().forEach((track) => track.stop());
  stream = null;
}

async function capturePhoto() {
  if (!stream) {
    setStatus("Start back camera first.");
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

  const focusPoint = await resolveFocusPoint(captureCanvas);
  const result = await compressToPhotoLimit(captureCanvas, focusPoint);
  if (!result) {
    setStatus("Could not fit 630x810 under 250 KB. Retry with plain background and even lighting.");
    return;
  }

  const fileName = `photo-mobile-${Date.now()}.jpg`;
  lastCaptureBlob = new File([result.blob], fileName, { type: "image/jpeg" });

  if (previewEl.src) {
    URL.revokeObjectURL(previewEl.src);
  }

  previewEl.src = URL.createObjectURL(lastCaptureBlob);
  outputSizeEl.textContent = `${formatKB(lastCaptureBlob.size)} (${PHOTO_WIDTH}x${PHOTO_HEIGHT})`;
  downloadBtn.disabled = false;
  retakeBtn.disabled = false;

  const centeringHint = focusPoint.faceCentered
    ? " Face auto-centered."
    : " Centered using face outline guide.";
  setStatus(`Captured photo as JPG in ${formatKB(lastCaptureBlob.size)}.${centeringHint}`, true);
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

function downloadPhoto() {
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

startBtn.addEventListener("click", startBackCamera);
captureBtn.addEventListener("click", capturePhoto);
retakeBtn.addEventListener("click", resetCapture);
downloadBtn.addEventListener("click", downloadPhoto);
window.addEventListener("beforeunload", stopCamera);
