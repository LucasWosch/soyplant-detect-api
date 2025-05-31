import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import models
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = "C:/Users/lucas37805/Documents/Projetos/soyplant-detect-api/src/vggAnnotation/soja_detector_model.keras"

model = models.load_model(MODEL_PATH, compile=False)

def detectar_soja_na_imagem(pil_image: Image.Image):
    """
    Recebe uma imagem PIL, detecta a soja e exibe a imagem com a bounding box.
    """
    # Converte PIL para numpy (OpenCV usa BGR)
    original = np.array(pil_image.convert("RGB"))
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]

    # Redimensiona e normaliza para o modelo
    img = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img_input = np.expand_dims(img, axis=0)

    # Faz predição
    pred = model.predict(img_input)[0]
    class_id, x, y, bw, bh = pred

    boxes = []

    if class_id > 0.5:
        x = int(x * w)
        y = int(y * h)
        bw = int(bw * w)
        bh = int(bh * h)

        x1 = max(0, x - bw // 2)
        y1 = max(0, y - bh // 2)
        x2 = min(w, x + bw // 2)
        y2 = min(h, y + bh // 2)

        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": float(round(class_id, 4))
        })

        # Desenha a bounding box na imagem
        boxed_img = original.copy()
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(boxed_img, f"Soja ({class_id:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Exibe a imagem com matplotlib (converte BGR -> RGB)
        plt.imshow(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box Detectada")
        plt.axis('off')
        plt.show()

    else:
        return {
            "pred": pred.tolist(),
            "msg": "Nenhum pé de soja detectado com confiança suficiente."
        }

    return {"pred": pred, "msg": "Nenhum pé de soja detectado com confiança suficiente.", "boxes": boxes}
