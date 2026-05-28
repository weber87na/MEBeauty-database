from pathlib import Path
from typing import Any

import base64
import cv2
import json
import numpy as np
import pandas as pd
import torch
import time
from io import BytesIO
from torch.utils.data import DataLoader
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, UnidentifiedImageError

from pytorch_mebeauty_dataset import MEBeauty, build_transform
from scut_model import load_scut_resnet18


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "pytorch_trained_models"
WEB_DIR = BASE_DIR / "webapp"
REFERENCE_SCORES = BASE_DIR / "scores" / "test_crop.csv"
PREDICTION_INSPECTION = MODEL_DIR / "prediction_inspection.csv"
RESULTS_DIR = BASE_DIR / "results"
HISTORY_LOG = RESULTS_DIR / "history.jsonl"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MODEL_SUFFIXES = {".pht", ".pth", ".pt"}
MIN_SCORE = 1.0
MAX_SCORE = 10.0
SCUT_MIN_SCORE = 1.0
SCUT_MAX_SCORE = 5.0
FACE_MARGIN = 0.25
VISIBILITY_THRESHOLD = 8.5
LOW_BEAUTY_THRESHOLD = 5.0
THREAT_THRESHOLD = 0.65
SURVIVAL_THRESHOLD = 0.65
SCORE_TRIGGER_THRESHOLD = 2
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
DEEPFACE_IMPORT_ERROR = None
EMOTION_LABELS_ZH = {
    "angry": "生氣",
    "disgust": "厭惡",
    "fear": "害怕",
    "happy": "開心",
    "sad": "難過",
    "surprise": "驚訝",
    "neutral": "自然",
}
EMOTION_SCORE_WEIGHTS = {
    "happy": 1.0,
    "neutral": 0.65,
    "surprise": 0.55,
    "sad": 0.25,
    "fear": 0.2,
    "angry": 0.1,
    "disgust": 0.0,
}


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


def get_deepface():
    global DEEPFACE_IMPORT_ERROR

    if DEEPFACE_IMPORT_ERROR is not None:
        return None

    try:
        from deepface import DeepFace
    except Exception as exc:
        DEEPFACE_IMPORT_ERROR = str(exc)
        return None

    return DeepFace


def emotion_score(emotions: dict[str, Any]) -> float:
    weighted_total = 0.0
    confidence_total = 0.0

    for name, confidence in emotions.items():
        value = float(confidence)
        weighted_total += value * EMOTION_SCORE_WEIGHTS.get(name, 0.0)
        confidence_total += value

    if confidence_total <= 0:
        return 0.0

    return min(max((weighted_total / confidence_total) * 10.0, 0.0), 10.0)


def threat_score(emotions: dict[str, Any] | None) -> float:
    if not emotions:
        return 0.0

    angry = float(emotions.get("angry", 0.0)) / 100.0
    disgust = float(emotions.get("disgust", 0.0)) / 100.0
    fear = float(emotions.get("fear", 0.0)) / 100.0
    surprise = float(emotions.get("surprise", 0.0)) / 100.0
    sad = float(emotions.get("sad", 0.0)) / 100.0

    score = angry * 1.0 + disgust * 0.85 + fear * 0.7 + surprise * 0.2 + sad * 0.15
    return round(min(max(score, 0.0), 1.0), 3)


def image_context_weight(image: Image.Image, face_box=None) -> float:
    gray = np.array(image.convert("L"))
    darkness = max(0.0, (105.0 - float(gray.mean())) / 105.0)

    proximity = 0.0
    if face_box is not None:
        _, _, face_width, face_height = face_box
        proximity = min(max((max(face_width, face_height) / max(image.size)) - 0.45, 0.0) / 0.35, 1.0)

    return round(min(max(darkness * 0.55 + proximity * 0.45, 0.0), 1.0), 3)


def normalized_beauty_score(score: float, score_scale: str) -> float:
    if score_scale == "1-5":
        return min(max(score * 2.0, 1.0), 10.0)
    return min(max(score, 1.0), 10.0)


def estimate_behavior(score: float, score_scale: str, emotion: dict[str, Any], face_detected: bool, env_weight: float, high_threshold: float):
    beauty = normalized_beauty_score(score, score_scale)
    threat = threat_score(emotion.get("emotions") if emotion.get("available") else None)
    survival_score = round(min(max(threat * 0.8 + env_weight * 0.2, 0.0), 1.0), 3)
    reasons = []
    trigger_score = 0
    mode = "preview"

    if not face_detected:
        reasons.append("未偵測到清楚人臉")
    elif survival_score >= SURVIVAL_THRESHOLD or threat >= THREAT_THRESHOLD:
        trigger_score += 3
        mode = "survival_beauty"
        reasons.append("威脅表情偏高，相機啟動求生美顏")
    elif beauty >= high_threshold:
        trigger_score += 2
        mode = "normal_beauty"
        reasons.append("顏值分數達高顏值門檻")
    elif beauty >= LOW_BEAUTY_THRESHOLD:
        trigger_score += 2
        mode = "headless_photo"
        reasons.append("顏值分數落在普通區間，觸發吐槽文字")
    else:
        reasons.append("分數未達正式效果門檻")

    if emotion.get("available"):
        reasons.append(f"主要表情：{emotion.get('dominant_label', emotion.get('dominant', '未知'))}")
    if env_weight >= 0.45:
        reasons.append("畫面偏暗或臉部距離太近")

    if trigger_score < SCORE_TRIGGER_THRESHOLD:
        mode = "preview"

    mode_labels = {
        "normal_beauty": "正常美照",
        "headless_photo": "吐槽模式",
        "survival_beauty": "畏懼美顏",
        "preview": "預覽模式",
    }

    return {
        "mode": mode,
        "label": mode_labels[mode],
        "reasons": reasons,
        "trigger_score": trigger_score,
        "beauty_score_normalized": round(beauty, 2),
        "threat_score": threat,
        "environment_weight": env_weight,
        "survival_score": survival_score,
        "thresholds": {
            "high_beauty": round(high_threshold, 1),
            "low_beauty": LOW_BEAUTY_THRESHOLD,
            "threat": THREAT_THRESHOLD,
            "survival": SURVIVAL_THRESHOLD,
        },
    }


def prepare_emotion_image(image: Image.Image, face_box=None) -> tuple[Image.Image, str]:
    if face_box is not None:
        emotion_image = crop_face_with_margin(image, face_box, FACE_MARGIN)
        source = "face-crop"
    else:
        emotion_image = image
        source = "full-image"

    emotion_image = ImageOps.autocontrast(emotion_image.convert("RGB"), cutoff=1)
    emotion_image = emotion_image.filter(ImageFilter.SHARPEN)

    min_side = min(emotion_image.size)
    if min_side and min_side < 224:
        scale = 224 / min_side
        resized_size = (
            max(1, int(emotion_image.width * scale)),
            max(1, int(emotion_image.height * scale)),
        )
        emotion_image = emotion_image.resize(resized_size, Image.Resampling.BICUBIC)

    return emotion_image, source


def analyze_face_emotion(image: Image.Image, face_box=None) -> dict[str, Any]:
    DeepFace = get_deepface()
    if DeepFace is None:
        return {
            "available": False,
            "error": DEEPFACE_IMPORT_ERROR or "DeepFace is not available.",
        }

    emotion_image, source = prepare_emotion_image(image, face_box)

    try:
        image_array = np.array(emotion_image)
        analysis = DeepFace.analyze(
            img_path=image_array,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else {}

    raw_emotions = analysis.get("emotion") or {}
    emotions = {
        name: round(float(raw_emotions.get(name, 0.0)), 2)
        for name in EMOTION_SCORE_WEIGHTS
    }
    dominant = str(analysis.get("dominant_emotion") or max(emotions, key=emotions.get, default="neutral"))
    score = emotion_score(emotions)

    return {
        "available": True,
        "score": round(score, 2),
        "score_scale": "0-10",
        "dominant": dominant,
        "dominant_label": EMOTION_LABELS_ZH.get(dominant, dominant),
        "confidence": emotions.get(dominant, 0.0),
        "emotions": emotions,
        "source": source,
        "preprocessing": ["face_crop" if face_box is not None else "full_image", "autocontrast", "sharpen"],
    }


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
    image.convert("RGB").save(buffer, format="JPEG", quality=96, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_from_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def load_ui_font(size: int, *, bold: bool = False):
    for font_path in [
        Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
        Path("C:/Windows/Fonts/msjhbd.ttc") if bold else Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def load_emoji_font(size: int):
    for font_path in [
        Path("C:/Windows/Fonts/seguiemj.ttf"),
        Path("C:/Windows/Fonts/seguisym.ttf"),
        Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    ]:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return load_ui_font(size, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_soft_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    *,
    fill=(255, 255, 255, 255),
    stroke_fill=(32, 32, 32, 210),
    stroke_width: int = 2,
) -> None:
    draw.text(xy, text, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)


def add_watermark(image: Image.Image, text: str, subtitle: str | None = None) -> Image.Image:
    result = image.convert("RGB").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = result.size
    font = load_ui_font(max(22, int(width * 0.042)), bold=True)
    small_font = load_ui_font(max(14, int(width * 0.026)))
    padding = max(16, int(width * 0.035))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    subtitle_bbox = draw.textbbox((0, 0), subtitle or "", font=small_font)
    box_width = max(text_bbox[2] - text_bbox[0], subtitle_bbox[2] - subtitle_bbox[0]) + padding * 2
    box_height = (text_bbox[3] - text_bbox[1]) + padding
    if subtitle:
        box_height += (subtitle_bbox[3] - subtitle_bbox[1]) + int(padding * 0.55)

    left = padding
    top = max(padding, height - box_height - padding)
    right = min(width - padding, left + box_width)
    bottom = min(height - padding, top + box_height)
    draw.rounded_rectangle((left, top, right, bottom), radius=max(8, padding // 2), fill=(16, 20, 26, 178))
    draw_soft_label(
        draw,
        (left + padding, top + padding // 2),
        text,
        font,
        fill=(255, 255, 255, 255),
        stroke_fill=(0, 0, 0, 180),
        stroke_width=max(1, int(width * 0.003)),
    )
    if subtitle:
        draw_soft_label(
            draw,
            (left + padding, top + padding // 2 + text_bbox[3] - text_bbox[1] + int(padding * 0.45)),
            subtitle,
            fill=(255, 250, 205, 255),
            font=small_font,
            stroke_fill=(0, 0, 0, 150),
            stroke_width=max(1, int(width * 0.002)),
        )

    return Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")


def draw_sticker(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    sticker: str,
    font,
    *,
    fill=(255, 255, 255, 245),
    stroke_fill=(0, 0, 0, 140),
) -> None:
    sticker_w, sticker_h = text_size(draw, sticker, font)
    draw.text(
        (int(center[0] - sticker_w / 2), int(center[1] - sticker_h / 2)),
        sticker,
        fill=fill,
        font=font,
        stroke_width=max(1, sticker_w // 28),
        stroke_fill=stroke_fill,
    )


def clamp_position(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def normalized_point(x: float, y: float, width: int, height: int) -> dict[str, float]:
    return {
        "x": round(min(max(x / width, 0.02), 0.98), 4),
        "y": round(min(max(y / height, 0.02), 0.98), 4),
    }


def build_emotion_overlay(image: Image.Image, face_box, emotion: dict[str, Any], mode: str, title: str, subtitle: str):
    width, height = image.size
    dominant = emotion.get("dominant") if emotion.get("available") else None
    if face_box is None:
        x = width * 0.5
        y = height * 0.22
        face_w = width * 0.28
        face_h = height * 0.28
    else:
        fx, fy, fw, fh = face_box
        x = fx + fw / 2
        y = fy
        face_w = fw
        face_h = fh

    items: list[dict[str, Any]] = []

    def sticker(text: str, px: float, py: float, size: float = 0.07) -> None:
        items.append({
            "type": "sticker",
            "text": text,
            "size": round(size, 4),
            **normalized_point(px, py, width, height),
        })

    def label(text: str, px: float, py: float, size: float = 0.035) -> None:
        items.append({
            "type": "label",
            "text": text,
            "size": round(size, 4),
            **normalized_point(px, py, width, height),
        })

    if dominant in {"angry", "disgust", "fear"} or mode == "survival_beauty":
        sticker_y = max(16, y - face_h * 0.22)
        sticker("👹", x - face_w * 0.62, sticker_y, 0.075)
        sticker("💢", x + face_w * 0.62, sticker_y, 0.06)
        label("相機求生中", x, max(18, sticker_y - face_h * 0.18), 0.034)
    elif dominant == "happy":
        sticker("✨", x - face_w * 0.65, y + face_h * 0.12, 0.055)
        sticker("💖", x + face_w * 0.65, y + face_h * 0.12, 0.055)
        sticker("😊", x + face_w * 0.58, y + face_h * 0.92, 0.052)
    elif dominant == "surprise":
        sticker("❗", x + face_w * 0.65, y + face_h * 0.02, 0.075)
        sticker("😳", x - face_w * 0.65, y + face_h * 0.08, 0.055)
    else:
        label("今天走自然派", x, max(18, y - face_h * 0.18), 0.034)

    return {
        "render": "dom",
        "items": items,
        "watermark": {
            "title": title,
            "subtitle": subtitle,
        },
    }


def add_emotion_decorations(image: Image.Image, face_box, emotion: dict[str, Any], mode: str) -> tuple[Image.Image, dict[str, Any]]:
    result = image.convert("RGB").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = result.size
    font = load_ui_font(max(18, int(width * 0.032)), bold=True)
    emoji_font = load_emoji_font(max(30, int(width * 0.065)))
    small_emoji_font = load_emoji_font(max(22, int(width * 0.045)))
    regions = []

    dominant = emotion.get("dominant") if emotion.get("available") else None
    if face_box is None:
        x = int(width * 0.5)
        y = int(height * 0.22)
        face_w = int(width * 0.28)
        face_h = int(height * 0.28)
    else:
        fx, fy, fw, fh = face_box
        x = int(fx + fw / 2)
        y = int(fy)
        face_w = fw
        face_h = fh

    if dominant in {"angry", "disgust", "fear"} or mode == "survival_beauty":
        sticker_y = max(16, y - int(face_h * 0.22))
        draw_sticker(draw, (clamp_position(int(x - face_w * 0.58), 24, width - 24), sticker_y), "👹", emoji_font)
        draw_sticker(draw, (clamp_position(int(x + face_w * 0.58), 24, width - 24), sticker_y), "💢", small_emoji_font)
        draw_soft_label(
            draw,
            (max(8, int(x - face_w * 0.5)), max(8, sticker_y - int(face_h * 0.18))),
            "相機求生中",
            font,
            fill=(255, 246, 120, 255),
            stroke_fill=(80, 24, 24, 220),
            stroke_width=max(2, int(width * 0.004)),
        )
        regions.extend(["emoji_oni", "emoji_anger", "survival_caption"])
    elif dominant == "happy":
        draw_sticker(draw, (clamp_position(int(x - face_w * 0.62), 24, width - 24), int(y + face_h * 0.12)), "✨", small_emoji_font)
        draw_sticker(draw, (clamp_position(int(x + face_w * 0.62), 24, width - 24), int(y + face_h * 0.12)), "💖", small_emoji_font)
        draw_sticker(draw, (clamp_position(int(x + face_w * 0.55), 24, width - 24), clamp_position(int(y + face_h * 0.92), 24, height - 24)), "😊", small_emoji_font)
        regions.extend(["emoji_sparkle", "emoji_heart", "emoji_smile"])
    elif dominant == "surprise":
        draw_sticker(draw, (clamp_position(int(x + face_w * 0.62), 24, width - 24), max(24, int(y + face_h * 0.02))), "❗", emoji_font)
        draw_sticker(draw, (clamp_position(int(x - face_w * 0.62), 24, width - 24), max(24, int(y + face_h * 0.08))), "😳", small_emoji_font)
        regions.extend(["emoji_surprise", "emoji_flushed"])
    else:
        draw_soft_label(
            draw,
            (max(8, int(x - face_w * 0.36)), max(8, int(y - face_h * 0.18))),
            "今天走自然派",
            font,
            fill=(255, 255, 255, 240),
            stroke_fill=(0, 0, 0, 180),
            stroke_width=max(2, int(width * 0.004)),
        )
        regions.extend(["neutral_caption"])

    decorated = Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")
    return decorated, {
        "applied": bool(regions),
        "method": "non-destructive-emotion-decorations",
        "regions": regions,
    }


def render_behavior_result(image: Image.Image, face_box, behavior: dict[str, Any], score: float, threshold: float):
    result_image, blur_applied, face_visible, protection = protected_result_image(image, face_box, score, threshold)
    return result_image, protection, face_visible, blur_applied


def render_decorated_result(image: Image.Image, face_box, behavior: dict[str, Any], emotion: dict[str, Any], score: float, threshold: float):
    base_image, blur_applied, face_visible, protection = protected_result_pil(image, face_box, score, threshold)
    mode = behavior["mode"]
    titles = {
        "normal_beauty": ("高顏值認證", "外貌協會 AI 相機"),
        "headless_photo": ("吐槽模式", "保留原圖，只追加文字"),
        "survival_beauty": ("畏懼美顏", "偵測到威脅表情，追加求生裝飾"),
        "preview": ("外貌協會 AI 相機", "保留原圖效果"),
    }
    title, subtitle = titles[mode]
    overlay = build_emotion_overlay(base_image, face_box, emotion, mode, title, subtitle)
    decoration = {
        "applied": bool(overlay["items"]),
        "method": "browser-native-dom-overlay",
        "regions": [item["text"] for item in overlay["items"]],
    }
    effect = {
        **protection,
        "decoration": decoration,
        "overlay": overlay,
        "method": protection.get("method", "none"),
        "non_destructive": True,
    }
    return image_data_url(base_image), effect, face_visible, blur_applied


def record_prediction_history(payload: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with HISTORY_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def protected_result_pil(image: Image.Image, face_box, score: float, threshold: float = VISIBILITY_THRESHOLD):
    details = {
        "applied": False,
        "method": "none",
        "landmarks_detected": False,
        "regions": [],
        "threshold": threshold,
    }

    if score >= threshold:
        return image.copy(), False, True, details

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

    return result, True, False, details


def protected_result_image(image: Image.Image, face_box, score: float, threshold: float = VISIBILITY_THRESHOLD):
    result, blur_applied, face_visible, details = protected_result_pil(image, face_box, score, threshold)
    return image_data_url(result), blur_applied, face_visible, details


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
    started_at = time.perf_counter()
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
    emotion = analyze_face_emotion(pil_image, prediction["face_box"])
    env_weight = image_context_weight(pil_image, prediction["face_box"])
    behavior = estimate_behavior(
        score,
        prediction["score_scale"],
        emotion,
        prediction["face_detected"],
        env_weight,
        threshold,
    )
    result_image, effect, face_visible, blur_applied = render_decorated_result(
        pil_image,
        prediction["face_box"],
        behavior,
        emotion,
        score,
        threshold,
    )
    latency_ms = round((time.perf_counter() - started_at) * 1000.0, 1)
    watermark_text = {
        "normal_beauty": "高顏值認證",
        "headless_photo": "吐槽模式",
        "survival_beauty": "畏懼美顏",
        "preview": "外貌協會 AI 相機",
    }[behavior["mode"]]

    history_item = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": resolved_model.name,
        "score": round(score, 2),
        "score_scale": prediction["score_scale"],
        "mode": behavior["mode"],
        "threat_score": behavior["threat_score"],
        "emotion": emotion.get("dominant_label") if emotion.get("available") else None,
        "latency_ms": latency_ms,
        "face_detected": prediction["face_detected"],
    }
    try:
        record_prediction_history(history_item)
    except OSError:
        history_item["log_error"] = "history write failed"

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
        "protection": effect,
        "effect_applied": bool(effect.get("applied")),
        "behavior": behavior,
        "watermark_text": watermark_text,
        "latency_ms": latency_ms,
        "emotion": emotion,
        "overlay": effect.get("overlay"),
        "result_image": result_image,
    }
