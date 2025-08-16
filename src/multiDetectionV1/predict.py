import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import models
from PIL import Image

IMG_SIZE = 224
N_BOXES = 10
MODEL_PATH = 'soja_detector_multibox.keras'

model = models.load_model("C:/Users/Gamer/PycharmProjects/soyplant-detect-api/src/multiDetectionV1/soja_detector_multibox.keras", compile=False)

def pil_to_bgr(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def detectar_soja_multibox(pil_image: Image.Image, conf_threshold=0.05, min_size=0.01):
    """
    Prediz até N_BOXES e desenha todas as caixas com conf >= conf_threshold.
    min_size é um filtro para caixas muito pequenas (normalizado).
    """
    original = pil_to_bgr(pil_image)
    h, w = original.shape[:2]

    # Prepara entrada
    img = cv2.resize(original, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    img = np.expand_dims(img, 0)

    # (1, N_BOXES, 5)
    preds = model.predict(img, verbose=0)[0]
    boxes_out = []
    boxed = original.copy()

    for i in range(preds.shape[0]):
        conf, x, y, bw, bh = preds[i]

        if conf < conf_threshold:
            continue
        if bw < min_size or bh < min_size:
            continue

        # Para pixels
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

        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(boxed, f"Soja ({conf:.2f})", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    # Exibição
    plt.imshow(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecções: {len(boxes_out)}")
    plt.axis('off')
    plt.show()

    return {"boxes": boxes_out, "raw": preds.tolist()}

