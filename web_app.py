from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from pytorch_mebeauty_dataset import build_transform


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "pytorch_trained_models"
WEB_DIR = BASE_DIR / "webapp"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MODEL_SUFFIXES = {".pht", ".pth", ".pt"}
MIN_SCORE = 1.0
MAX_SCORE = 10.0

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
IMAGE_TRANSFORM = build_transform(False)
MODEL_CACHE: dict[str, Any] = {}


app = FastAPI(title="MEBeauty Score Web App")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def list_model_paths() -> list[Path]:
    if not MODEL_DIR.exists():
        return []

    return sorted(
        [path for path in MODEL_DIR.iterdir() if path.suffix.lower() in ALLOWED_MODEL_SUFFIXES],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_model_path(model_path: str) -> Path:
    path = (BASE_DIR / model_path).resolve()
    model_root = MODEL_DIR.resolve()

    if not path.is_file():
        raise HTTPException(status_code=400, detail="Model file does not exist.")
    if path.suffix.lower() not in ALLOWED_MODEL_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported model file type.")
    if model_root not in path.parents:
        raise HTTPException(status_code=400, detail="Model must be inside pytorch_trained_models.")

    return path


def load_model(path: Path):
    cache_key = str(path)
    if cache_key not in MODEL_CACHE:
        model = torch.load(path, map_location=DEVICE, weights_only=False)
        model = model.to(DEVICE)
        model.eval()
        MODEL_CACHE[cache_key] = model

    return MODEL_CACHE[cache_key]


def image_to_tensor(upload: UploadFile):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported images: JPG, PNG, WEBP, BMP.")

    try:
        image = Image.open(upload.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc

    tensor = IMAGE_TRANSFORM(image).float()
    return tensor.view(1, *tensor.shape).to(DEVICE)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/models")
def models():
    model_paths = list_model_paths()
    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models": [
            {
                "name": path.name,
                "path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
                "modified": path.stat().st_mtime,
            }
            for path in model_paths
        ],
    }


@app.post("/api/predict")
def predict(model_path: str = Form(...), image: UploadFile = File(...)):
    resolved_model = resolve_model_path(model_path)
    tensor = image_to_tensor(image)
    model = load_model(resolved_model)

    with torch.no_grad():
        raw_score = float(model(tensor).reshape(-1).item())
        score = min(max(raw_score, MIN_SCORE), MAX_SCORE)

    return {
        "score": round(score, 2),
        "raw_score": round(raw_score, 4),
        "model": resolved_model.name,
        "device": str(DEVICE),
    }
