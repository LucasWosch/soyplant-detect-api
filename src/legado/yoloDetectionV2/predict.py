import base64
import contextlib
import io

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from typing import Tuple

# ======= CONFIG =======
MODEL_PATH = r"/runs/detect/train5/weights/best.pt"  # ajuste se necessário

# Seleção automática de device
_HAS_CUDA = torch.cuda.is_available()
_DEVICE_AUTO = 0 if _HAS_CUDA else "cpu"
_HALF_AUTO = bool(_HAS_CUDA)  # usa FP16 apenas na GPU

# Carrega o modelo 1x (Ultralytics gerencia os pesos no device na hora do predict)
_model = YOLO(MODEL_PATH)
CLASS_NAMES = _model.names  # dict {id: name}

# ======= PRÉ-PROCESSAMENTO (somente VERDE) =======
def mask_green_hsv(
    bgr: np.ndarray,
    low: Tuple[int, int, int] = (35, 25, 25),
    high: Tuple[int, int, int] = (90, 255, 255),
) -> np.ndarray:
    """Máscara de VERDE em HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(low, dtype=np.uint8)
    upper = np.array(high, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def refine_mask(mask: np.ndarray, k: int = 3, iterations: int = 1) -> np.ndarray:
    """Limpa ruídos e fecha buracos pequenos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask

def remove_small_blobs(mask: np.ndarray, min_area: int = 60) -> np.ndarray:
    """Remove componentes muito pequenos (pontinhos)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):  # 0 = background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def keep_colors_and_black_rest(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mantém cores onde mask==255 e zera (preto) no resto."""
    result = np.zeros_like(bgr)
    result[mask > 0] = bgr[mask > 0]
    return result

def preprocess_green_only(
    bgr: np.ndarray,
    green_hsv=(35, 25, 25, 90, 255, 255),
    min_area=60,
    morph_k=3,
    morph_iter=1,
) -> np.ndarray:
    """Pipeline: máscara verde -> refino -> remove ruído -> aplica."""
    g_low, g_high = green_hsv[:3], green_hsv[3:]
    m_green = mask_green_hsv(bgr, g_low, g_high)
    m_green = refine_mask(m_green, k=morph_k, iterations=morph_iter)
    m_green = remove_small_blobs(m_green, min_area=min_area)
    out = keep_colors_and_black_rest(bgr, m_green)
    return out

# ======= HELPERS =======
def _encode_bgr_to_data_url(bgr_img, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem anotada.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _pil_to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _safe_name(cid: int):
    if isinstance(CLASS_NAMES, dict):
        return CLASS_NAMES.get(cid, str(cid))
    return str(cid)

# ======= PREDICT (com pré-processamento verde) =======
def predict_yolo_V2(
    pil_image: Image.Image,
    conf_threshold: float = 0.1,
    imgsz: int = 640,
    device_override=None,   # 0, 1, "cpu" etc. Se None, usa auto
    half: bool = None,      # Se None, usa HALF_AUTO. Passe True/False para forçar
    # parâmetros do pré-processamento:
    green_hsv=(35, 25, 25, 90, 255, 255),
    min_area: int = 60,
    morph_k: int = 3,
    morph_iter: int = 1,
):
    """
    Retorna:
      - boxes: [{x1,y1,x2,y2,confidence,class_id,name}]
      - boxes_norm: [{x,y,w,h,confidence,class_id,name}]
      - raw: lista com extras (pixels/norm/área)
      - image_base64: preview com boxes desenhados (sobre a imagem PRÉ-PROCESSADA)
      - arquitetura/model/model_lines: descrição textual do modelo
    """
    device = _DEVICE_AUTO if device_override is None else device_override
    use_half = _HALF_AUTO if half is None else (bool(half) and device != "cpu")

    # PIL -> BGR
    bgr_original = _pil_to_bgr(pil_image)

    # PRÉ-PROCESSAMENTO (somente VERDE)
    bgr = preprocess_green_only(
        bgr_original,
        green_hsv=green_hsv,
        min_area=min_area,
        morph_k=morph_k,
        morph_iter=morph_iter,
    )

    H, W = bgr.shape[:2]

    # Inferência
    results = _model.predict(
        source=[bgr],           # imagem PRÉ-PROCESSADA em memória
        device=device,          # 0 para GPU, "cpu" para CPU
        imgsz=imgsz,
        conf=conf_threshold,
        half=use_half,          # FP16 em GPU
        verbose=False
    )

    annotated = bgr.copy()
    boxes_pixels, boxes_norm, raw = [], [], []

    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(float)  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = xyxy
            x1 = max(0.0, min(W - 1.0, x1))
            y1 = max(0.0, min(H - 1.0, y1))
            x2 = max(0.0, min(W - 1.0, x2))
            y2 = max(0.0, min(H - 1.0, y2))

            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # normalizados [0..1]
            x = cx / W
            y = cy / H
            wn = w / W
            hn = h / H

            conf = float(b.conf[0].detach().cpu().item()) if b.conf is not None else 0.0
            cls_id = int(b.cls[0].detach().cpu().item()) if b.cls is not None else 0
            name = _safe_name(cls_id)

            boxes_pixels.append({
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "name": name
            })

            boxes_norm.append({
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "w": round(float(wn), 6),
                "h": round(float(hn), 6),
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "name": name
            })

            raw.append({
                "pixels": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                           "w": float(w), "h": float(h), "cx": float(cx), "cy": float(cy)},
                "norm": {"x": float(x), "y": float(y), "w": float(wn), "h": float(hn)},
                "confidence": float(conf),
                "class_id": int(cls_id),
                "name": name,
                "area_px": float(w * h)
            })

            # desenha na imagem PRÉ-PROCESSADA
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{name} {conf:.2f}",
                (int(x1), max(15, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                2
            )

    # ordena por confiança
    boxes_pixels.sort(key=lambda b: b["confidence"], reverse=True)
    boxes_norm.sort(key=lambda b: b["confidence"], reverse=True)
    raw.sort(key=lambda b: b["confidence"], reverse=True)

    # Captura arquitetura como string (model.info() imprime no stdout)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _model.info()
    arquitetura_str = buf.getvalue()

    model_str = str(_model.model)
    model_lines = model_str.splitlines()

    return {
        "device_used": device,
        "half": use_half,
        "boxes": boxes_pixels,
        "boxes_norm": boxes_norm,
        "raw": raw,
        "image_base64": _encode_bgr_to_data_url(annotated, quality=90),  # preview da IMAGEM PRÉ-PROCESSADA
        "arquitetura": arquitetura_str,   # string
        "model": model_str,               # string (com \n)
        "model_lines": model_lines,       # lista linha a linha
        "preprocess": {
            "green_hsv": list(green_hsv),
            "min_area": int(min_area),
            "morph_k": int(morph_k),
            "morph_iter": int(morph_iter)
        }
    }
