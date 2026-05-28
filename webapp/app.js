const modelSelect = document.querySelector("#modelSelect");
const thresholdSelect = document.querySelector("#thresholdSelect");
const deviceLabel = document.querySelector("#device");
const form = document.querySelector("#predictForm");
const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const imageOverlay = document.querySelector("#imageOverlay");
const emptyState = document.querySelector("#emptyState");
const scoreLabel = document.querySelector("#score");
const relativeRank = document.querySelector("#relativeRank");
const cropDetails = document.querySelector("#cropDetails");
const toggleDetailsButton = document.querySelector("#toggleDetailsButton");
const detailsContent = document.querySelector("#detailsContent");
const message = document.querySelector("#message");
const submitButton = document.querySelector("#submitButton");
const clearButton = document.querySelector("#clearButton");
const downloadButton = document.querySelector("#downloadButton");
const shareButton = document.querySelector("#shareButton");
const cameraButton = document.querySelector("#cameraButton");
const captureButton = document.querySelector("#captureButton");
const cameraVideo = document.querySelector("#cameraVideo");
const cameraCanvas = document.querySelector("#cameraCanvas");
const behaviorMode = document.querySelector("#behaviorMode");
const behaviorReasons = document.querySelector("#behaviorReasons");
const bestBeauty = document.querySelector("#bestBeauty");
const bestThreat = document.querySelector("#bestThreat");
const sessionHistory = document.querySelector("#sessionHistory");

let selectedFile = null;
let cameraStream = null;
let detailsExpanded = false;
let latestResult = null;
let sessionResults = [];

function setMessage(text) {
  message.textContent = text || "";
}

function setLoading(isLoading) {
  clearButton.disabled = isLoading;
  downloadButton.disabled = isLoading || !latestResult?.result_image;
  shareButton.disabled = isLoading || !latestResult;
}

function resetResult() {
  scoreLabel.textContent = "--";
  relativeRank.textContent = "相對等級 --";
  behaviorMode.textContent = "尚未判定";
  behaviorReasons.textContent = "拍照或選擇圖片後顯示觸發原因。";
  latestResult = null;
  imageOverlay.innerHTML = "";
  imageOverlay.style.display = "none";
  downloadButton.disabled = true;
  shareButton.disabled = true;
  
  // Reset details panel to collapsed state
  detailsExpanded = false;
  detailsContent.style.display = "none";
  cropDetails.classList.remove("expanded");
  toggleDetailsButton.setAttribute("aria-expanded", "false");
  
  const detailsTitle = document.querySelector(".detailsTitle");
  detailsTitle.textContent = "Crop details";
  detailsContent.innerHTML = `<div class="detailsEmpty">尚未產生詳細分數</div>`;
}

function syncImageOverlayBox() {
  if (!latestResult?.overlay || previewImage.style.display === "none") {
    imageOverlay.style.display = "none";
    return;
  }

  const stageRect = previewImage.parentElement.getBoundingClientRect();
  const imageRect = previewImage.getBoundingClientRect();
  imageOverlay.style.left = `${imageRect.left - stageRect.left}px`;
  imageOverlay.style.top = `${imageRect.top - stageRect.top}px`;
  imageOverlay.style.width = `${imageRect.width}px`;
  imageOverlay.style.height = `${imageRect.height}px`;
  imageOverlay.querySelectorAll(".overlaySticker, .overlayLabel").forEach((node) => {
    const size = Number(node.dataset.size || 0.055);
    node.style.fontSize = `${Math.max(18, imageRect.width * size)}px`;
  });
  imageOverlay.querySelectorAll(".overlayWatermark").forEach((node) => {
    node.style.fontSize = `${Math.max(16, Math.min(28, imageRect.width * 0.048))}px`;
  });
  imageOverlay.style.display = "block";
}

function renderImageOverlay(data) {
  const overlay = data.overlay || data.protection?.overlay;
  imageOverlay.innerHTML = "";

  if (!overlay) {
    imageOverlay.style.display = "none";
    return;
  }

  if (overlay.watermark?.title) {
    const watermark = document.createElement("div");
    watermark.className = "overlayWatermark";
    watermark.textContent = overlay.watermark.title;
    if (overlay.watermark.subtitle) {
      const subtitle = document.createElement("small");
      subtitle.textContent = overlay.watermark.subtitle;
      watermark.appendChild(subtitle);
    }
    imageOverlay.appendChild(watermark);
  }

  (overlay.items || []).forEach((item) => {
    const node = document.createElement("div");
    node.className = item.type === "label" ? "overlayLabel" : "overlaySticker";
    node.textContent = item.text;
    node.dataset.size = item.size || 0.06;
    node.style.setProperty("--x", item.x);
    node.style.setProperty("--y", item.y);
    imageOverlay.appendChild(node);
  });

  requestAnimationFrame(syncImageOverlayBox);
}

function renderBehavior(data) {
  const behavior = data.behavior || {};
  behaviorMode.textContent = behavior.mode === "headless_photo" ? "吐槽模式" : (behavior.label || "預覽模式");
  const reasons = Array.isArray(behavior.reasons) && behavior.reasons.length
    ? behavior.reasons.join(" · ")
    : "沒有明確觸發原因";
  const threat = Number(behavior.threat_score || 0).toFixed(2);
  const survival = Number(behavior.survival_score || 0).toFixed(2);
  behaviorReasons.textContent = `${reasons} · 威脅指數 ${threat} · 求生指數 ${survival}`;
}

function updateSessionHistory(data) {
  const behavior = data.behavior || {};
  sessionResults.unshift({
    mode: behavior.mode === "headless_photo" ? "吐槽模式" : (behavior.label || behavior.mode || "預覽模式"),
    score: Number(data.score || 0),
    threat: Number(behavior.threat_score || 0),
    emotion: data.emotion?.dominant_label || data.emotion?.dominant || "--",
  });
  sessionResults = sessionResults.slice(0, 10);

  const highestBeauty = sessionResults.reduce((best, item) => Math.max(best, item.score), 0);
  const highestThreat = sessionResults.reduce((best, item) => Math.max(best, item.threat), 0);
  bestBeauty.textContent = highestBeauty ? highestBeauty.toFixed(2) : "--";
  bestThreat.textContent = highestThreat ? highestThreat.toFixed(2) : "--";

  sessionHistory.innerHTML = sessionResults.map((item) => `
    <div class="historyItem">
      <span>${item.mode}</span>
      <strong>${item.score.toFixed(2)}</strong>
      <em>${item.emotion} · threat ${item.threat.toFixed(2)}</em>
    </div>
  `).join("");
}

function shareText(data) {
  const behavior = data.behavior || {};
  const emotion = data.emotion?.dominant_label || data.emotion?.dominant || "--";
  const modeLabel = behavior.mode === "headless_photo" ? "吐槽模式" : (behavior.label || "預覽模式");
  return `外貌協會 AI 相機：${modeLabel}｜顏值 ${Number(data.score).toFixed(2)}｜表情 ${emotion}｜威脅指數 ${Number(behavior.threat_score || 0).toFixed(2)}`;
}

function drawRoundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

async function downloadCompositedResult() {
  if (!latestResult?.result_image) return;

  const source = await loadImage(latestResult.result_image);
  const canvas = document.createElement("canvas");
  canvas.width = source.naturalWidth || source.width;
  canvas.height = source.naturalHeight || source.height;
  const context = canvas.getContext("2d");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);

  const overlay = latestResult.overlay || latestResult.protection?.overlay;
  if (overlay) {
    (overlay.items || []).forEach((item) => {
      const x = item.x * canvas.width;
      const y = item.y * canvas.height;
      const size = Math.max(18, canvas.width * (item.size || 0.055));
      if (item.type === "label") {
        context.font = `900 ${size}px "Noto Sans TC", "Microsoft JhengHei", sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        const metrics = context.measureText(item.text);
        const padX = size * 0.55;
        const padY = size * 0.35;
        drawRoundedRect(context, x - metrics.width / 2 - padX, y - size / 2 - padY, metrics.width + padX * 2, size + padY * 2, 10);
        context.fillStyle = "rgba(16, 20, 26, 0.72)";
        context.fill();
        context.lineWidth = Math.max(2, size * 0.08);
        context.strokeStyle = "rgba(0, 0, 0, 0.65)";
        context.strokeText(item.text, x, y);
        context.fillStyle = "#ffffff";
        context.fillText(item.text, x, y);
      } else {
        context.font = `${size}px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(item.text, x, y);
      }
    });

    if (overlay.watermark?.title) {
      const titleSize = Math.max(18, Math.min(32, canvas.width * 0.05));
      const subtitleSize = titleSize * 0.58;
      context.font = `900 ${titleSize}px "Noto Sans TC", "Microsoft JhengHei", sans-serif`;
      const titleWidth = context.measureText(overlay.watermark.title).width;
      context.font = `700 ${subtitleSize}px "Noto Sans TC", "Microsoft JhengHei", sans-serif`;
      const subtitleWidth = overlay.watermark.subtitle ? context.measureText(overlay.watermark.subtitle).width : 0;
      const boxWidth = Math.max(titleWidth, subtitleWidth) + titleSize * 1.2;
      const boxHeight = overlay.watermark.subtitle ? titleSize * 2.05 : titleSize * 1.45;
      const left = canvas.width * 0.04;
      const top = canvas.height - boxHeight - canvas.width * 0.04;
      drawRoundedRect(context, left, top, boxWidth, boxHeight, 12);
      context.fillStyle = "rgba(16, 20, 26, 0.72)";
      context.fill();
      context.textAlign = "left";
      context.textBaseline = "top";
      context.font = `900 ${titleSize}px "Noto Sans TC", "Microsoft JhengHei", sans-serif`;
      context.fillStyle = "#ffffff";
      context.fillText(overlay.watermark.title, left + titleSize * 0.55, top + titleSize * 0.28);
      if (overlay.watermark.subtitle) {
        context.font = `700 ${subtitleSize}px "Noto Sans TC", "Microsoft JhengHei", sans-serif`;
        context.fillStyle = "#fff4b0";
        context.fillText(overlay.watermark.subtitle, left + titleSize * 0.55, top + titleSize * 1.25);
      }
    }
  }

  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = `mebeauty-${latestResult.behavior?.mode || "result"}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function renderCropDetails(data) {
  const crops = data.crops || [];
  const mode = data.inference_mode || (crops.length > 1 ? "multi-crop-average" : "single-crop");
  const protection = data.protection || {};
  const emotion = data.emotion || {};
  const protectionMethod = protection.method || (data.blur_applied ? "legacy-blur" : "none");
  const protectedRegions = Array.isArray(protection.regions) && protection.regions.length
    ? protection.regions.join(", ")
    : "--";
  const protectionRow = `
    <div class="detailRow protection">
      <span>face protection</span>
      <span>${protectionMethod}</span>
      <strong>${protection.applied ? "on" : "off"}</strong>
      <em>${protection.landmarks_detected ? protectedRegions : "no landmarks"}</em>
    </div>
  `;
  const emotionRow = emotion.available ? `
    <div class="detailRow emotion">
      <span>表情親和分數</span>
      <span>${emotion.dominant_label || emotion.dominant || "--"}</span>
      <strong>${Number(emotion.score).toFixed(2)}</strong>
      <em>${Number(emotion.confidence || 0).toFixed(1)}% 模型信心</em>
    </div>
  ` : `
    <div class="detailRow emotion">
      <span>表情親和分數</span>
      <span>DeepFace</span>
      <strong>--</strong>
      <em>${emotion.error || "unavailable"}</em>
    </div>
  `;
  const emotionBreakdown = emotion.available && emotion.emotions ? `
    <div class="emotionGrid">
      ${Object.entries(emotion.emotions).map(([name, value]) => `
        <div class="emotionItem">
          <span>${name}</span>
          <strong>${Number(value).toFixed(1)}%</strong>
        </div>
      `).join("")}
    </div>
  ` : "";
  const behavior = data.behavior || {};
  const behaviorRows = `
    <div class="detailRow behavior">
      <span>decision</span>
      <span>${behavior.label || behavior.mode || "--"}</span>
      <strong>${Number(behavior.trigger_score || 0).toFixed(0)}</strong>
      <em>${Number(behavior.threat_score || 0).toFixed(2)} threat</em>
    </div>
    <div class="detailRow behavior">
      <span>effect</span>
      <span>${data.protection?.method || "--"}</span>
      <strong>${data.effect_applied ? "on" : "off"}</strong>
      <em>${Number(data.latency_ms || 0).toFixed(1)} ms</em>
    </div>
  `;

  const detailsContent = document.getElementById("detailsContent");
  const detailsTitle = document.querySelector(".detailsTitle");
  
  if (!crops.length) {
    detailsTitle.textContent = `Crop details · ${mode}`;
    detailsContent.innerHTML = `
      <div class="detailRow summary">
        <span>score</span>
        <span>${data.score_scale || "--"}</span>
        <strong>${Number(data.score).toFixed(2)}</strong>
        <em>raw ${Number(data.raw_score ?? data.score).toFixed(4)}</em>
      </div>
      ${behaviorRows}
      ${protectionRow}
      ${emotionRow}
      ${emotionBreakdown}
    `;
    return;
  }

  const rows = crops.map((crop) => {
    const margin = crop.margin === null ? "full" : crop.margin.toFixed(2);
    return `
      <div class="detailRow">
        <span>${crop.name}</span>
        <span>margin ${margin}</span>
        <strong>${Number(crop.score).toFixed(2)}</strong>
        <em>raw ${Number(crop.raw_score).toFixed(4)}</em>
      </div>
    `;
  }).join("");

  detailsTitle.textContent = `Crop details · ${mode}`;
  detailsContent.innerHTML = `
    ${rows}
    <div class="detailRow summary">
      <span>average</span>
      <span>${data.score_scale}</span>
      <strong>${Number(data.score).toFixed(2)}</strong>
      <em>raw ${Number(data.raw_score).toFixed(4)}</em>
    </div>
    ${behaviorRows}
    ${protectionRow}
    ${emotionRow}
    ${emotionBreakdown}
  `;
}

function setPreviewFromFile(file) {
  selectedFile = file;
  resetResult();
  setMessage("");

  if (!file) {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    imageOverlay.innerHTML = "";
    imageOverlay.style.display = "none";
    emptyState.style.display = "flex";
    emptyState.textContent = "尚未選擇圖片";
    setLoading(false);
    return;
  }

  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  imageOverlay.innerHTML = "";
  imageOverlay.style.display = "none";
  emptyState.style.display = "flex";
  emptyState.textContent = "圖片已選擇，測試後顯示結果";
  setLoading(false);
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraVideo.style.display = "none";
  captureButton.disabled = true;
  cameraButton.textContent = "開啟相機";
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();

  deviceLabel.textContent = data.device || "unknown";
  modelSelect.innerHTML = "";

  if (!data.models.length) {
    const option = document.createElement("option");
    option.textContent = "找不到模型";
    option.value = "";
    modelSelect.appendChild(option);
    modelSelect.disabled = true;
    setMessage("請先訓練模型，或把 .pht/.pth/.pt 放進 pytorch_trained_models。");
    setLoading(false);
    return;
  }

  data.models.filter((model) => !model.name.includes("scut")).forEach((model) => {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = `${model.name} · MEBeauty 1-10`;
    modelSelect.appendChild(option);
  });

  if (!modelSelect.options.length) {
    const option = document.createElement("option");
    option.textContent = "找不到可用模型";
    option.value = "";
    modelSelect.appendChild(option);
    modelSelect.disabled = true;
    setMessage("請先放入 MEBeauty 模型，例如 best_model.pht。");
    setLoading(false);
    return;
  }

  modelSelect.disabled = false;
  setLoading(false);
}

async function autoSubmitPrediction() {
  if (!selectedFile || !modelSelect.value) return;
  
  setMessage("");
  resetResult();
  
  const threshold = parseFloat(thresholdSelect.value);
  
  const formData = new FormData();
  formData.append("model_path", modelSelect.value);
  formData.append("image", selectedFile);
  formData.append("threshold", threshold.toString());
  
  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || "推論失敗。");
    }
    
    scoreLabel.textContent = Number(data.score).toFixed(2);
    latestResult = data;
    if (data.result_image) {
      previewImage.onload = () => renderImageOverlay(data);
      previewImage.src = data.result_image;
      previewImage.style.display = "block";
      emptyState.style.display = "none";
      renderImageOverlay(data);
    }
    if (data.relative && data.relative.percentile !== null) {
      relativeRank.textContent = `${data.relative.band} · 高於 ${data.relative.percentile}% 測試集樣本 · 約前 ${data.relative.top_percent}%`;
    } else if (data.score_scale) {
      relativeRank.textContent = `${data.relative.band} · 分數尺度 ${data.score_scale}`;
    } else {
      relativeRank.textContent = "相對等級無法計算";
    }
    renderBehavior(data);
    updateSessionHistory(data);
    renderCropDetails(data);
    const thresholdValue = data.threshold !== undefined ? Number(data.threshold).toFixed(1) : thresholdSelect.value;
    const emotionText = data.emotion?.available ? `表情親和分數 ${Number(data.emotion.score).toFixed(2)}。` : "";
    if (data.behavior?.mode && data.behavior.mode !== "preview") {
      const modeLabel = data.behavior.mode === "headless_photo" ? "吐槽模式" : data.behavior.label;
      setMessage(`${modeLabel} 已套用。${emotionText}`);
    } else if (data.face_visible) {
      setMessage(`分數達 ${thresholdValue} 以上，顯示原圖人臉。${emotionText}`);
    } else if (data.blur_applied) {
      const method = data.protection?.method || "feature protection";
      setMessage(`分數低於 ${thresholdValue}，五官已自動淡出（${method}）。${emotionText}`);
    } else {
      setMessage(data.face_detected ? "已自動偵測並裁切最大的人臉。" : "未偵測到清楚人臉，已使用整張圖片。");
    }
  } catch (error) {
    setMessage(error.message);
  } finally {
    setLoading(false);
  }
}

imageInput.addEventListener("change", async () => {
  setPreviewFromFile(imageInput.files[0] || null);
  if (imageInput.files[0] && modelSelect.value) {
    setTimeout(autoSubmitPrediction, 100);
  }
});

thresholdSelect.addEventListener("change", () => {
  if (selectedFile && modelSelect.value) {
    autoSubmitPrediction();
  }
});

cameraButton.addEventListener("click", async () => {
  if (cameraStream) {
    stopCamera();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.style.display = "block";
    captureButton.disabled = false;
    cameraButton.textContent = "關閉相機";
    setMessage("");
  } catch (error) {
    setMessage("無法開啟相機，請確認瀏覽器權限。");
  }
});

captureButton.addEventListener("click", () => {
  if (!cameraStream) {
    return;
  }

  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const context = cameraCanvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

  cameraCanvas.toBlob(async (blob) => {
    if (!blob) {
      setMessage("拍照失敗，請再試一次。");
      return;
    }

    const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
    imageInput.value = "";
    setPreviewFromFile(file);
    stopCamera();
    
    if (modelSelect.value) {
      setTimeout(autoSubmitPrediction, 100);
    }
  }, "image/jpeg", 0.92);
});

clearButton.addEventListener("click", () => {
  stopCamera();
  selectedFile = null;
  imageInput.value = "";
  setPreviewFromFile(null);
  setMessage("");
});

downloadButton.addEventListener("click", () => {
  downloadCompositedResult().catch(() => {
    setMessage("下載合成失敗，請再試一次。");
  });
});

shareButton.addEventListener("click", async () => {
  if (!latestResult) return;

  const text = shareText(latestResult);
  try {
    await navigator.clipboard.writeText(text);
    setMessage("已複製分享文字。");
  } catch (error) {
    setMessage(text);
  }
});

toggleDetailsButton.addEventListener("click", (event) => {
  event.preventDefault();
  detailsExpanded = !detailsExpanded;
  detailsContent.style.display = detailsExpanded ? "block" : "none";
  cropDetails.classList.toggle("expanded", detailsExpanded);
  toggleDetailsButton.setAttribute("aria-expanded", detailsExpanded);
});

window.addEventListener("resize", syncImageOverlayBox);

form.addEventListener("submit", (event) => {
  event.preventDefault();
});

loadModels().catch((error) => {
  deviceLabel.textContent = "error";
  setMessage(error.message);
  setLoading(false);
});
