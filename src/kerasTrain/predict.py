import os
import cv2
import numpy as np
from PIL import Image
from keras import models

# Caminho para o modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, './soyplant_cnn_model.h5')

# Carregamento do modelo
model = models.load_model(model_path)

def apply_sobel_pil(image: Image.Image) -> np.ndarray:
    """
    Aplica o filtro de Sobel à imagem PIL convertida para escala de cinza.
    """
    gray = np.array(image.convert("L"))  # Converte para escala de cinza
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))  # Garante faixa 0-255
    return sobel

def predict_image(image: Image.Image) -> float:
    """
    Realiza a predição da imagem usando o modelo treinado.
    """
    sobel_image = apply_sobel_pil(image)
    resized = cv2.resize(sobel_image, (128, 128))
    normalized = resized.astype("float32") / 255.0
    input_image = normalized.reshape(1, 128, 128, 1)
    prediction = model.predict(input_image, verbose=0)
    return float(prediction[0][0])
