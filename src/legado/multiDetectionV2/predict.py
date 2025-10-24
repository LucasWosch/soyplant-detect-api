# multiDetectionV1/predict.py
import base64
import numpy as np
import cv2
from keras import models
from PIL import Image

IMG_SIZE = 224  # deve coincidir com o treino (usamos apenas para o resize da entrada)
# Caminho do modelo treinado v2 (ajuste conforme seu ambiente)
MODEL_PATH = r"/src/legado/multiDetectionV2/soja_detector_multibox_v2.keras"

# Carrega uma única vez
model = models.load_model(MODEL_PATH, compile=False)

def _infer_n_boxes_from_model(m):
    """
    Tenta inferir N_BOXES de m.output_shape: esperado (None, N_BOXES, 5).
    """
    out_shape = getattr(m, "output_shape", None)
    if isinstance(out_shape, tuple) and len(out_shape) == 3 and out_shape[2] == 5:
        return int(out_shape[1])
    # fallback (caso algum wrapper mude o shape)
    try:
        # chama uma vez com dummy para descobrir o shape
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out = m.predict(dummy, verbose=0)
        return int(out.shape[1])
    except Exception:
        # último recurso: padrão
        return 5

N_BOXES = _infer_n_boxes_from_model(model)

def pil_to_bgr(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _encode_bgr_to_data_url(bgr_img, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem anotada.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def detectar_soja_multibox_V2(
    pil_image: Image.Image,
    conf_threshold: float = 0.1,
    min_size: float = 0.01
):
    """
    Prediz até N_BOXES e retorna:
      - boxes: lista de caixas em pixels (após filtros) -> [{x1,y1,x2,y2,confidence}]
      - boxes_norm: lista de caixas normalizadas (após filtros) -> [{x,y,w,h,confidence}]
      - raw: TODAS as previsões com nomes [conf, x, y, w, h] normalizados [0..1] (sem filtro)
      - image_base64: imagem anotada (JPEG) em data URL
    """
    original_bgr = pil_to_bgr(pil_image)
    H, W = original_bgr.shape[:2]

    # Prepara entrada (o treino v2 normaliza por 255 e usa IMG_SIZE)
    img = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img = np.expand_dims(img, 0)

    # Saída: (1, N_BOXES, 5) => [conf, x, y, w, h] normalizados
    preds = model.predict(img, verbose=0)[0]

    boxes_pixels = []
    boxes_norm_filtered = []
    raw_named = []
    annotated = original_bgr.copy()

    for i in range(preds.shape[0]):
        conf, x, y, bw, bh = map(float, preds[i])

        # Sempre guarda no raw (sem filtro), com clamp para [0,1]
        x = _clamp01(x)
        y = _clamp01(y)
        bw = _clamp01(bw)
        bh = _clamp01(bh)
        conf = float(conf)

        raw_named.append({
            "conf": conf,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh
        })

        # Filtros para reportar/desenhar
        if conf < conf_threshold:
            continue
        if bw < min_size or bh < min_size:
            continue

        # Normalizado -> pixels (cx,cy,w,h)
        cx = int(x * W)
        cy = int(y * H)
        pw = int(bw * W)
        ph = int(bh * H)

        x1 = max(0, cx - pw // 2)
        y1 = max(0, cy - ph // 2)
        x2 = min(W, cx + pw // 2)
        y2 = min(H, cy + ph // 2)

        boxes_pixels.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "confidence": round(conf, 4)
        })

        boxes_norm_filtered.append({
            "x": x, "y": y, "w": bw, "h": bh,
            "confidence": round(conf, 4)
        })

        # Desenha
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            f"Soja ({conf:.2f})",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2
        )

    # (Opcional) ordenar por confiança
    boxes_pixels.sort(key=lambda b: b["confidence"], reverse=True)
    boxes_norm_filtered.sort(key=lambda b: b["confidence"], reverse=True)

    # Gera imagem anotada como data URL
    data_url = _encode_bgr_to_data_url(annotated, quality=90)

    return {
        "n_boxes_model": int(N_BOXES),     # informativo
        "boxes": boxes_pixels,             # em pixels (pós-filtro)
        "boxes_norm": boxes_norm_filtered, # normalizados (pós-filtro)
        "raw": raw_named,                  # todas as saídas normalizadas, sem filtro
        "image_base64": data_url
    }
