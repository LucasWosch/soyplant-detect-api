# yolo_predict.py
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# ======= CONFIG =======
MODEL_PATH = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/runs/detect/train8/weights/best.pt"  # ajuste se necessário

# Seleção automática de device
_HAS_CUDA = torch.cuda.is_available()
_DEVICE_AUTO = 0 if _HAS_CUDA else "cpu"
_HALF_AUTO = bool(_HAS_CUDA)  # usa FP16 apenas na GPU

# Carrega o modelo 1x (Ultralytics gerencia os pesos no device na hora do predict)
_model = YOLO(MODEL_PATH)
CLASS_NAMES = _model.names  # dict {id: name}

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

# ======= PREDICT =======
def predict_yolo(
    pil_image: Image.Image,
    conf_threshold: float = 0.1,
    imgsz: int = 640,
    device_override=None,   # use 0, 1, "cpu", etc. Se None, usa auto
    half: bool = None       # Se None, usa HALF_AUTO. Passe True/False para forçar
):
    """
    Retorna:
      - boxes: [{x1,y1,x2,y2,confidence,class_id,name}]
      - boxes_norm: [{x,y,w,h,confidence,class_id,name}]
      - raw: lista com extras (pixels/norm/área)
      - image_base64: preview com boxes desenhados
    """
    device = _DEVICE_AUTO if device_override is None else device_override
    use_half = _HALF_AUTO if half is None else (bool(half) and device != "cpu")

    bgr = _pil_to_bgr(pil_image)
    H, W = bgr.shape[:2]

    # Inferência (Ultralytics move os tensores internamente conforme o 'device')
    results = _model.predict(
        source=[bgr],           # imagem em memória
        device=device,          # 0 para GPU, "cpu" para CPU
        imgsz=imgsz,
        conf=conf_threshold,
        half=use_half,          # FP16 em GPU
        verbose=False
    )

    annotated = bgr.copy()
    boxes_pixels = []
    boxes_norm = []
    raw = []

    for r in results:
        if r.boxes is None:
            continue
        # r.boxes.xyxy/conf/cls são tensores (no device), convertemos de forma segura
        for b in r.boxes:
            # xyxy em pixels
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(float)  # [x1, y1, x2, y2]
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

            # conf e classe
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

            # desenha
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

    model_str = str(_model.model)
    model_lines = model_str.splitlines()

    return {
        "device_used": device,
        "half": use_half,
        "boxes": boxes_pixels,
        "boxes_norm": boxes_norm,
        "raw": raw,
        "image_base64": _encode_bgr_to_data_url(annotated, quality=90),
        "arquitetura": _model.info(),  # string
        "model": model_str,  # string (com \n)
        "model_lines": model_lines  # <<< lista de linhas (mais legível no JSON)
    }

