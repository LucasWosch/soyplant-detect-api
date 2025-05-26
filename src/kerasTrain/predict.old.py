import tensorflow as tf
import numpy as np
import cv2
import os

# Caminho da pasta de imagens
current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(current_dir, '../../data/sobel_images')
model_path = os.path.join(current_dir, '../../soyplant_cnn_model.h5')

# Carregamento do modelo
model = tf.keras.models.load_model(model_path)

# Função para aplicar filtro de Sobel
def apply_sobel(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel

# Percorrer todas as imagens da pasta e fazer predições
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(image_folder, filename)
        try:
            sobel_image = apply_sobel(image_path)
            resized = cv2.resize(sobel_image, (128, 128))
            normalized = resized / 255.0
            input_image = normalized.reshape(1, 128, 128, 1)
            prediction = model.predict(input_image)
            print(f"{filename} => Predição: {prediction[0][0]:.4f}")
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
