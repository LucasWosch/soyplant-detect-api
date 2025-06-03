import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import models
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = "C:/Users/lucas37805/Documents/Projetos/soyplant-detect-api/src/vggAnnotation/soja_segmentation_model.keras"

model = models.load_model(MODEL_PATH, compile=False)

def detectar_poligonos_soja(pil_image: Image.Image, threshold=0.5):
    """
    Recebe uma imagem PIL, prediz a máscara de soja e desenha os contornos como polígonos.
    """
    # Converte para BGR (OpenCV) e guarda dimensões originais
    original = np.array(pil_image.convert("RGB"))
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    h_orig, w_orig = original_bgr.shape[:2]

    # Redimensiona para o modelo
    resized = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))
    input_img = resized.astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Faz a predição da máscara
    pred_mask = model.predict(input_img)[0, :, :, 0]

    # Aplica threshold para binarizar
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # Redimensiona de volta à escala original
    mask_resized = cv2.resize(binary_mask, (w_orig, h_orig))

    # Encontra contornos
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            "num_poligonos": 0,
            "msg": "Nenhum polígono detectado com confiança suficiente."
        }

    # Desenha os contornos na imagem
    output = original_bgr.copy()
    for contour in contours:
        cv2.polylines(output, [contour], isClosed=True, color=(255, 255, 255), thickness=2)

    # Exibe a imagem
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Polígonos Detectados")
    plt.axis('off')
    plt.show()

    return {
        "num_poligonos": len(contours),
        "msg": "Segmentação realizada com sucesso."
    }
