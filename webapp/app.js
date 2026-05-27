const modelSelect = document.querySelector("#modelSelect");
const thresholdSelect = document.querySelector("#thresholdSelect");
const deviceLabel = document.querySelector("#device");
const form = document.querySelector("#predictForm");
const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const emptyState = document.querySelector("#emptyState");
const scoreLabel = document.querySelector("#score");
const relativeRank = document.querySelector("#relativeRank");
const cropDetails = document.querySelector("#cropDetails");
const toggleDetailsButton = document.querySelector("#toggleDetailsButton");
const detailsContent = document.querySelector("#detailsContent");
const message = document.querySelector("#message");
const submitButton = document.querySelector("#submitButton");
const clearButton = document.querySelector("#clearButton");
const cameraButton = document.querySelector("#cameraButton");
const captureButton = document.querySelector("#captureButton");
const cameraVideo = document.querySelector("#cameraVideo");
const cameraCanvas = document.querySelector("#cameraCanvas");

let selectedFile = null;
let cameraStream = null;
let detailsExpanded = false;

function setMessage(text) {
  message.textContent = text || "";
}

function setLoading(isLoading) {
  clearButton.disabled = isLoading;
}

function resetResult() {
  scoreLabel.textContent = "--";
  relativeRank.textContent = "相對等級 --";
  
  // Reset details panel to collapsed state
  detailsExpanded = false;
  detailsContent.style.display = "none";
  cropDetails.classList.remove("expanded");
  toggleDetailsButton.setAttribute("aria-expanded", "false");
  
  const detailsTitle = document.querySelector(".detailsTitle");
  detailsTitle.textContent = "Crop details";
  detailsContent.innerHTML = `<div class="detailsEmpty">尚未產生詳細分數</div>`;
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
    emptyState.style.display = "flex";
    emptyState.textContent = "尚未選擇圖片";
    setLoading(false);
    return;
  }

  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
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
    if (data.result_image) {
      previewImage.src = data.result_image;
      previewImage.style.display = "block";
      emptyState.style.display = "none";
    }
    if (data.relative && data.relative.percentile !== null) {
      relativeRank.textContent = `${data.relative.band} · 高於 ${data.relative.percentile}% 測試集樣本 · 約前 ${data.relative.top_percent}%`;
    } else if (data.score_scale) {
      relativeRank.textContent = `${data.relative.band} · 分數尺度 ${data.score_scale}`;
    } else {
      relativeRank.textContent = "相對等級無法計算";
    }
    renderCropDetails(data);
    const thresholdValue = data.threshold !== undefined ? Number(data.threshold).toFixed(1) : thresholdSelect.value;
    if (data.face_visible) {
      const emotionText = data.emotion?.available ? `表情親和分數 ${Number(data.emotion.score).toFixed(2)}，${data.emotion.dominant_label || data.emotion.dominant}。` : "";
      setMessage(`分數達 ${thresholdValue} 以上，顯示原圖人臉。${emotionText}`);
    } else if (data.blur_applied) {
      const method = data.protection?.method || "feature protection";
      const emotionText = data.emotion?.available ? `表情親和分數 ${Number(data.emotion.score).toFixed(2)}。` : "";
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

toggleDetailsButton.addEventListener("click", (event) => {
  event.preventDefault();
  detailsExpanded = !detailsExpanded;
  detailsContent.style.display = detailsExpanded ? "block" : "none";
  cropDetails.classList.toggle("expanded", detailsExpanded);
  toggleDetailsButton.setAttribute("aria-expanded", detailsExpanded);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
});

loadModels().catch((error) => {
  deviceLabel.textContent = "error";
  setMessage(error.message);
  setLoading(false);
});
