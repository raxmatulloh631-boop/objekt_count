import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global model (loaded once)
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        _model = YOLO(settings.MODEL_NAME)
        # Set classes for open-vocabulary (will be updated per request)
        logger.info("Model loaded successfully")
    return _model


def count_objects(image_path: str, target_object: str, conf_threshold: float = None) -> dict:
    """
    Open-vocabulary object counting using YOLO-World.
    Only counts the target_object specified by user.
    """
    start_time = time.time()
    conf = conf_threshold or settings.CONFIDENCE_THRESHOLD

    model = get_model()

    # Set the target class for open-vocabulary detection
    # YOLO-World accepts text prompts
    model.set_classes([target_object])

    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
        imgsz=1280,  # higher resolution helps small objects
    )

    result = results[0]
    boxes = result.boxes

    detections = []
    confidences = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append({
                "label": target_object,
                "confidence": round(conf_score, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
            })
            confidences.append(conf_score)

    processing_time = int((time.time() - start_time) * 1000)
    avg_conf = float(np.mean(confidences)) if confidences else None

    # Create visualization
    processed_path = draw_detections(image_path, detections, target_object)

    warning = None
    if detections and avg_conf and avg_conf < 0.5:
        warning = "Ayrim obyektlar past confidence ga ega. Tekshirish tavsiya etiladi."
    elif len(detections) == 0:
        warning = f"'{target_object}' topilmadi. Boshqa rasm yoki aniqroq nom bilan urinib ko'ring."

    return {
        "detections": detections,
        "count": len(detections),
        "processing_time_ms": processing_time,
        "average_confidence": round(avg_conf, 4) if avg_conf else None,
        "processed_image_path": processed_path,
        "model_name": settings.MODEL_NAME,
        "warning": warning,
    }


def draw_detections(image_path: str, detections: list, target_object: str) -> str:
    """Draw boxes and numbers on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det.get("confidence", 0)

        # Box color (green)
        color = (34, 197, 94)  # green-500

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        label = f"{idx}"
        bbox = draw.textbbox((x1, y1 - 22), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 22), label, fill="white", font=font)

        # Small confidence
        conf_text = f"{int(conf * 100)}%"
        draw.text((x1 + 2, y2 + 2), conf_text, fill=color, font=small_font)

    # Save
    filename = Path(image_path).stem + "_processed.jpg"
    out_path = Path(settings.PROCESSED_DIR) / filename
    img.save(out_path, "JPEG", quality=90)

    return str(out_path)
