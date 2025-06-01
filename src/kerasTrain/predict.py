import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Caminho para o modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'soyplant_cnn_model.h5')

# Carregamento do modelo
model = load_model(model_path)

# Tamanho da entrada esperada pelo modelo
IMG_SIZE = (128, 128)

def apply_sobel_pil(image: Image.Image) -> np.ndarray:
    """
    Aplica o filtro de Sobel a uma imagem PIL convertida para escala de cinza.
    """
    gray = np.array(image.convert("L"))  # Convertido para escala de cinza
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))  # Garante faixa entre 0-255
    return sobel

def predict_image(image: Image.Image) -> float:
    """
    Realiza a predição da imagem usando o modelo treinado.
    """
    sobel_image = apply_sobel_pil(image)
    resized = cv2.resize(sobel_image, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0

    # Redimensiona para (1, altura, largura, canais)
    input_image = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    # Realiza predição
    prediction = model.predict(input_image, verbose=0)

    # Retorna a predição como float
    return float(prediction[0][0])
