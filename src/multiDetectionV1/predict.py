# multiDetectionV1/predict.py
import base64
import numpy as np
import cv2
from keras import models
from PIL import Image

IMG_SIZE = 224
N_BOXES = 10

# ajuste o caminho conforme seu ambiente
MODEL_PATH = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/src/multiDetectionV1/soja_detector_multibox.keras"
model = models.load_model(MODEL_PATH, compile=False)

def pil_to_bgr(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _encode_bgr_to_data_url(bgr_img, quality: int = 90) -> str:
    # codifica como JPEG e retorna data URL base64
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem anotada.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def detectar_soja_multibox(pil_image: Image.Image, conf_threshold=0.1, min_size=0.01):
    """
    Prediz até N_BOXES e retorna:
      - boxes: lista de caixas em pixels (após filtros)
      - raw: todas as previsões com nomes [conf, x, y, w, h] normalizados [0..1]
      - image_base64: imagem anotada (JPEG) em data URL
    """
    original = pil_to_bgr(pil_image)
    h, w = original.shape[:2]

    # Prepara entrada
    img = cv2.resize(original, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img = np.expand_dims(img, 0)

    # (1, N_BOXES, 5)
    preds = model.predict(img, verbose=0)[0]
    boxes_out = []
    raw_named = []
    boxed = original.copy()

    for i in range(preds.shape[0]):
        conf, x, y, bw, bh = preds[i]

        # sempre guardar no raw com nomes
        raw_named.append({
            "conf": float(conf),
            "x": float(x),
            "y": float(y),
            "w": float(bw),
            "h": float(bh)
        })

        # filtros apenas para desenhar/reportar em 'boxes'
        if conf < conf_threshold:
            continue
        if bw < min_size or bh < min_size:
            continue

        # normalizado -> pixels
        cx = int(x * w)
        cy = int(y * h)
        pw = int(bw * w)
        ph = int(bh * h)

        x1 = max(0, cx - pw // 2)
        y1 = max(0, cy - ph // 2)
        x2 = min(w, cx + pw // 2)
        y2 = min(h, cy + ph // 2)

        boxes_out.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "confidence": float(round(conf, 4))
        })

        # desenha
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            boxed,
            f"Soja ({conf:.2f})",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2
        )

    # gera imagem anotada como data URL
    data_url = _encode_bgr_to_data_url(boxed, quality=90)

    return {
        "boxes": boxes_out,
        "raw": raw_named,
        "image_base64": data_url
    }
