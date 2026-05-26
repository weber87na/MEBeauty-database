from pathlib import Path
from typing import Any

import base64
import cv2
import numpy as np
import pandas as pd
import torch
from io import BytesIO
from torch.utils.data import DataLoader
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from pytorch_mebeauty_dataset import MEBeauty, build_transform
from scut_model import load_scut_resnet18


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "pytorch_trained_models"
WEB_DIR = BASE_DIR / "webapp"
REFERENCE_SCORES = BASE_DIR / "scores" / "test_crop.csv"
PREDICTION_INSPECTION = MODEL_DIR / "prediction_inspection.csv"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MODEL_SUFFIXES = {".pht", ".pth", ".pt"}
MIN_SCORE = 1.0
MAX_SCORE = 10.0
SCUT_MIN_SCORE = 1.0
SCUT_MAX_SCORE = 5.0
FACE_MARGIN = 0.25
VISIBILITY_THRESHOLD = 8.5
CROP_MARGINS = [
    ("tight", 0.15),
    ("normal", 0.25),
    ("wide", 0.35),
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
IMAGE_TRANSFORM = build_transform(False)
MODEL_CACHE: dict[str, Any] = {}
REFERENCE_PREDICTIONS_CACHE: dict[str, list[float]] = {}
FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
FACE_KEYPOINT_DETECTOR = None
FACE_KEYPOINT_IMPORT_ERROR = None


app = FastAPI(title="高顏值照相機")
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
        if path.name == "scut_resnet18_py3.pth":
            model = load_scut_resnet18(path, DEVICE)
        else:
            model = torch.load(path, map_location=DEVICE, weights_only=False)
        model = model.to(DEVICE)
        model.eval()
        MODEL_CACHE[cache_key] = model

    return MODEL_CACHE[cache_key]


def score_band(percentile: float) -> str:
    if percentile >= 90:
        return "Top tier"
    if percentile >= 75:
        return "High"
    if percentile >= 50:
        return "Above average"
    if percentile >= 25:
        return "Average"
    return "Low"


def load_reference_predictions(model_path: Path) -> list[float]:
    cache_key = str(model_path.resolve())
    if cache_key in REFERENCE_PREDICTIONS_CACHE:
        return REFERENCE_PREDICTIONS_CACHE[cache_key]

    if PREDICTION_INSPECTION.exists() and model_path.name == "best_model.pht":
        predictions = pd.read_csv(PREDICTION_INSPECTION)["prediction"].dropna().astype(float).tolist()
    else:
        model = load_model(model_path)
        dataset = MEBeauty("", str(REFERENCE_SCORES), str(REFERENCE_SCORES), train=False, transform=IMAGE_TRANSFORM)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        predictions = []
        with torch.no_grad():
            for xb, _ in dataloader:
                xb = xb.to(DEVICE)
                batch_predictions = model(xb).reshape(-1).detach().cpu().tolist()
                predictions.extend(float(value) for value in batch_predictions)

    REFERENCE_PREDICTIONS_CACHE[cache_key] = sorted(predictions)
    return REFERENCE_PREDICTIONS_CACHE[cache_key]


def relative_rank(score: float, model_path: Path) -> dict[str, Any]:
    if model_path.name == "scut_resnet18_py3.pth":
        return {
            "percentile": None,
            "top_percent": None,
            "band": "SCUT-FBP5500 scale",
        }

    predictions = load_reference_predictions(model_path)
    if not predictions:
        return {"percentile": None, "top_percent": None, "band": "Unknown"}

    below_or_equal = sum(value <= score for value in predictions)
    percentile = 100.0 * below_or_equal / len(predictions)
    top_percent = max(0.0, 100.0 - percentile)

    return {
        "percentile": round(percentile, 1),
        "top_percent": round(top_percent, 1),
        "band": score_band(percentile),
    }


def clamp_score(raw_score: float, model_path: Path) -> tuple[float, str]:
    if model_path.name == "scut_resnet18_py3.pth":
        return min(max(raw_score, SCUT_MIN_SCORE), SCUT_MAX_SCORE), "1-5"
    return min(max(raw_score, MIN_SCORE), MAX_SCORE), "1-10"


def detect_largest_face_box(image: Image.Image):
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(width), int(height)


def get_face_keypoint_detector():
    global FACE_KEYPOINT_DETECTOR, FACE_KEYPOINT_IMPORT_ERROR

    if FACE_KEYPOINT_DETECTOR is not None:
        return FACE_KEYPOINT_DETECTOR
    if FACE_KEYPOINT_IMPORT_ERROR is not None:
        return None

    try:
        from mtcnn import MTCNN
    except Exception as exc:  # pragma: no cover - depends on optional local wheel
        FACE_KEYPOINT_IMPORT_ERROR = str(exc)
        return None

    FACE_KEYPOINT_DETECTOR = MTCNN()
    return FACE_KEYPOINT_DETECTOR


def detect_face_keypoints(image: Image.Image):
    detector = get_face_keypoint_detector()
    if detector is None:
        return None

    image_rgb = np.ascontiguousarray(np.array(image))
    detections = detector.detect_faces(image_rgb)
    if not detections:
        return None

    detection = max(detections, key=lambda item: item["box"][2] * item["box"][3])
    x, y, width, height = detection["box"]
    x = max(int(x), 0)
    y = max(int(y), 0)
    width = max(int(width), 1)
    height = max(int(height), 1)

    keypoints = {
        name: (int(point[0]), int(point[1]))
        for name, point in detection.get("keypoints", {}).items()
    }

    return {
        "box": (x, y, width, height),
        "keypoints": keypoints,
        "confidence": float(detection.get("confidence", 0.0)),
    }


def draw_feature_ellipse(mask, center, axes, region_name, regions):
    if center is None:
        return

    center = (int(center[0]), int(center[1]))
    axes = (max(3, int(axes[0])), max(3, int(axes[1])))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    regions.append(region_name)


def facial_feature_mask(image: Image.Image, detection: dict[str, Any]):
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)
    regions = []
    face_x, face_y, face_width, face_height = detection["box"]
    face_size = max(face_width, face_height, 1)
    keypoints = detection["keypoints"]

    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    nose = keypoints.get("nose")
    mouth_left = keypoints.get("mouth_left")
    mouth_right = keypoints.get("mouth_right")

    eye_axes = (face_width * 0.13, face_height * 0.075)
    brow_offset = int(face_height * 0.085)
    brow_axes = (face_width * 0.14, face_height * 0.06)
    nose_axes = (face_width * 0.13, face_height * 0.17)

    draw_feature_ellipse(mask, left_eye, eye_axes, "left_eye", regions)
    draw_feature_ellipse(mask, right_eye, eye_axes, "right_eye", regions)
    if left_eye:
        draw_feature_ellipse(mask, (left_eye[0], left_eye[1] - brow_offset), brow_axes, "left_brow", regions)
    if right_eye:
        draw_feature_ellipse(mask, (right_eye[0], right_eye[1] - brow_offset), brow_axes, "right_brow", regions)
    draw_feature_ellipse(mask, nose, nose_axes, "nose", regions)

    if mouth_left and mouth_right:
        mouth_center = (
            (mouth_left[0] + mouth_right[0]) // 2,
            (mouth_left[1] + mouth_right[1]) // 2,
        )
        mouth_width = max(abs(mouth_right[0] - mouth_left[0]), int(face_width * 0.18))
        draw_feature_ellipse(
            mask,
            mouth_center,
            (mouth_width * 0.72, face_height * 0.09),
            "mouth",
            regions,
        )

    if not regions:
        return None, []

    expand = max(7, int(face_size * 0.045))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(5, expand * 2 + 1), max(5, expand * 2 + 1)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    return mask, regions


def remove_facial_features_with_landmarks(image: Image.Image):
    detection = detect_face_keypoints(image)
    if detection is None:
        return None, {
            "method": "fallback",
            "landmarks_detected": False,
            "regions": [],
        }

    hard_mask, regions = facial_feature_mask(image, detection)
    if hard_mask is None:
        return None, {
            "method": "fallback",
            "landmarks_detected": True,
            "regions": [],
        }

    image_rgb = np.array(image)
    _, _, face_width, face_height = detection["box"]
    face_size = max(face_width, face_height, 1)
    inpaint_radius = max(5, int(face_size * 0.03))
    feather_radius = max(19, int(face_size * 0.09))
    if feather_radius % 2 == 0:
        feather_radius += 1

    inpainted = cv2.inpaint(image_rgb, hard_mask, inpaint_radius, cv2.INPAINT_TELEA)
    smoothed = cv2.bilateralFilter(inpainted, d=15, sigmaColor=55, sigmaSpace=55)
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=max(2, face_size * 0.012))

    soft_mask = cv2.GaussianBlur(hard_mask, (feather_radius, feather_radius), 0)
    alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None]
    alpha = np.clip(alpha * 1.25, 0.0, 1.0)
    result = image_rgb.astype(np.float32) * (1.0 - alpha) + smoothed.astype(np.float32) * alpha

    noise = np.random.default_rng(7).normal(0, 1.2, result.shape)
    texture_alpha = (hard_mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    result = result + noise * texture_alpha

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {
        "method": "mtcnn-keypoint-inpaint",
        "landmarks_detected": True,
        "regions": regions,
    }


def crop_face_with_margin(image: Image.Image, face_box, margin: float) -> Image.Image:
    x, y, width, height = face_box
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    left = max(x - margin_x, 0)
    top = max(y - margin_y, 0)
    right = min(x + width + margin_x, image.width)
    bottom = min(y + height + margin_y, image.height)

    return image.crop((left, top, right, bottom))


def face_region_with_margin(image: Image.Image, face_box, margin: float):
    x, y, width, height = face_box
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    left = max(x - margin_x, 0)
    top = max(y - margin_y, 0)
    right = min(x + width + margin_x, image.width)
    bottom = min(y + height + margin_y, image.height)

    return left, top, right, bottom


def facial_features_region(image: Image.Image, face_box):
    x, y, width, height = face_box
    left = int(x + width * 0.18)
    top = int(y + height * 0.22)
    right = int(x + width * 0.82)
    bottom = int(y + height * 0.78)

    return (
        max(left, 0),
        max(top, 0),
        min(right, image.width),
        min(bottom, image.height),
    )


def blank_facial_features(image: Image.Image, face_box):
    result = image.copy()
    region = facial_features_region(result, face_box)
    features = result.crop(region)

    x, y, width, height = face_box
    skin_region = (
        max(int(x + width * 0.18), 0),
        max(int(y + height * 0.72), 0),
        min(int(x + width * 0.82), image.width),
        min(int(y + height * 0.96), image.height),
    )
    skin_sample = result.crop(skin_region)
    skin_pixels = np.array(skin_sample).reshape(-1, 3)
    skin_color = tuple(int(value) for value in np.median(skin_pixels, axis=0))

    blank = Image.new("RGB", features.size, skin_color)
    blank_array = np.array(blank).astype(np.float32)
    noise = np.random.default_rng(7).normal(0, 2.0, blank_array.shape)
    blank = Image.fromarray(np.clip(blank_array + noise, 0, 255).astype(np.uint8))
    blank = blank.filter(ImageFilter.GaussianBlur(radius=14))

    mask = Image.new("L", features.size, 0)
    draw = ImageDraw.Draw(mask)
    inset_x = max(1, int(features.width * 0.01))
    inset_y = max(1, int(features.height * 0.01))
    draw.ellipse(
        (inset_x, inset_y, features.width - inset_x, features.height - inset_y),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(10, int(min(features.size) * 0.16))))

    mask_array = np.array(mask).astype(np.float32)
    mask_array = np.clip(mask_array * 1.35, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask_array)

    result.paste(blank, region, mask)
    return result


def image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def protected_result_image(image: Image.Image, face_box, score: float, threshold: float = VISIBILITY_THRESHOLD):
    details = {
        "applied": False,
        "method": "none",
        "landmarks_detected": False,
        "regions": [],
        "threshold": threshold,
    }

    if score >= threshold:
        return image_data_url(image), False, True, details

    result = image.copy()
    if face_box is None:
        result = result.filter(ImageFilter.GaussianBlur(radius=18))
        details.update({
            "applied": True,
            "method": "full-image-blur",
        })
    else:
        landmark_result, landmark_details = remove_facial_features_with_landmarks(result)
        if landmark_result is not None:
            result = landmark_result
            details.update(landmark_details)
            details["applied"] = True
        else:
            result = blank_facial_features(result, face_box)
            details.update(landmark_details)
            details["applied"] = True
            details["method"] = "face-box-oval-fallback"

    return image_data_url(result), True, False, details


def image_to_tensor(image: Image.Image):
    tensor = IMAGE_TRANSFORM(image).float()
    return tensor.view(1, *tensor.shape).to(DEVICE)


def uploaded_image(upload: UploadFile):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported images: JPG, PNG, WEBP, BMP.")

    try:
        image = Image.open(upload.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc

    return image


def predict_crops(model, image: Image.Image, model_path: Path):
    face_box = detect_largest_face_box(image)
    face_detected = face_box is not None
    crop_specs = []

    if face_detected:
        for name, margin in CROP_MARGINS:
            crop_specs.append((name, margin, crop_face_with_margin(image, face_box, margin)))
        inference_mode = "3-crop-average"
    else:
        crop_specs.append(("full image", None, image))
        inference_mode = "full-image"

    crop_results = []
    raw_scores = []
    with torch.no_grad():
        for name, margin, crop in crop_specs:
            tensor = image_to_tensor(crop)
            raw_score = float(model(tensor).reshape(-1).item())
            score, _ = clamp_score(raw_score, model_path)
            raw_scores.append(raw_score)
            crop_results.append({
                "name": name,
                "margin": margin,
                "raw_score": round(raw_score, 4),
                "score": round(score, 2),
            })

    raw_score = sum(raw_scores) / len(raw_scores)
    score, score_scale = clamp_score(raw_score, model_path)

    return {
        "score": score,
        "score_scale": score_scale,
        "raw_score": raw_score,
        "face_detected": face_detected,
        "face_box": face_box,
        "inference_mode": inference_mode,
        "crops": crop_results,
    }


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
def predict(model_path: str = Form(...), image: UploadFile = File(...), threshold: float = Form(default=8.5)):
    resolved_model = resolve_model_path(model_path)
    pil_image = uploaded_image(image)
    model = load_model(resolved_model)
    
    # 确保 threshold 是有效的浮点数
    try:
        threshold = float(threshold)
    except (ValueError, TypeError):
        threshold = 8.5
    
    # 限制 threshold 在合理范围内
    threshold = max(0.0, min(10.0, threshold))

    prediction = predict_crops(model, pil_image, resolved_model)
    score = prediction["score"]
    raw_score = prediction["raw_score"]
    rank = relative_rank(raw_score, resolved_model)
    result_image, blur_applied, face_visible, protection = protected_result_image(
        pil_image,
        prediction["face_box"],
        score,
        threshold,
    )

    return {
        "score": round(score, 2),
        "raw_score": round(raw_score, 4),
        "relative": rank,
        "model": resolved_model.name,
        "score_scale": prediction["score_scale"],
        "device": str(DEVICE),
        "face_detected": prediction["face_detected"],
        "inference_mode": prediction["inference_mode"],
        "crops": prediction["crops"],
        "threshold": round(threshold, 1),
        "face_visible": face_visible,
        "blur_applied": blur_applied,
        "protection": protection,
        "result_image": result_image,
    }
