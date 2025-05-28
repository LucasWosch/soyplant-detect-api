import os

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image

# Carregar modelo treinado
current_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(current_dir, "modelo_cnn_soja.h5")
modelo = load_model(modelo_path)

# Dicionário de classes
classes = {0: "nao_soja", 1: "soja"}

def prever_com_cnn(imagem_pil):
    # Redimensionar e normalizar a imagem
    img = imagem_pil.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    pred = modelo.predict(img_array)[0][0]
    classe = "soja" if pred >= 0.5 else "nao_soja"

    return {
        "classe": classe,
        "probabilidade": round(pred * 100, 2) if classe == "soja" else round((1 - pred) * 100, 2)
    }

def prever_quantidade_cnn(imagem_pil):
    img = imagem_pil.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)[0][0]
    return {
        "quantidade_estimativa": round(pred),
        "valor_continuo": float(pred)
    }