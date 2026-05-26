const modelSelect = document.querySelector("#modelSelect");
const deviceLabel = document.querySelector("#device");
const form = document.querySelector("#predictForm");
const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const emptyState = document.querySelector("#emptyState");
const scoreLabel = document.querySelector("#score");
const message = document.querySelector("#message");
const submitButton = document.querySelector("#submitButton");

function setMessage(text) {
  message.textContent = text || "";
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading || !modelSelect.value;
  submitButton.textContent = isLoading ? "測試中..." : "測試分數";
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

  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.name;
    modelSelect.appendChild(option);
  });

  modelSelect.disabled = false;
  setLoading(false);
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  scoreLabel.textContent = "--";
  setMessage("");

  if (!file) {
    previewImage.style.display = "none";
    emptyState.style.display = "flex";
    return;
  }

  previewImage.src = URL.createObjectURL(file);
  previewImage.style.display = "block";
  emptyState.style.display = "none";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("");
  scoreLabel.textContent = "--";

  const file = imageInput.files[0];
  if (!file) {
    setMessage("請先選擇圖片。");
    return;
  }

  const formData = new FormData();
  formData.append("model_path", modelSelect.value);
  formData.append("image", file);

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
  } catch (error) {
    setMessage(error.message);
  } finally {
    setLoading(false);
  }
});

loadModels().catch((error) => {
  deviceLabel.textContent = "error";
  setMessage(error.message);
  setLoading(false);
});
