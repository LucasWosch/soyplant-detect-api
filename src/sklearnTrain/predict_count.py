import os
import joblib
import numpy as np
from full_analysis import analisar_todos

current_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(current_dir, "modelo_soja_regressao.pkl")
modelo = joblib.load(modelo_path)

def prever_quantidade_soja(imagem_pil):
    # Extrair features da imagem
    features = analisar_todos(imagem_pil)

    # Organizar os dados no formato que o modelo espera
    X = np.array([[features["shi_tomasi"], features["harris"], features["contornos_verdes"]]])

    # Fazer predição
    pred = modelo.predict(X)[0]
    pred_ajustado = round(pred)

    return {
        "quantidade_estimada": pred_ajustado,
        "shi_tomasi": features["shi_tomasi"],
        "harris": features["harris"],
        "contornos_verdes": features["contornos_verdes"]
    }
