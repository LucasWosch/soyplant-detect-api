import os

import joblib
import numpy as np
from full_analysis import analisar_todos

# Carregar o modelo de classificação salvo
current_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(current_dir, "modelo_soja_classificacao.pkl")
modelo = joblib.load(modelo_path)

def prever_se_soja(imagem_pil):
    # Extrair features da imagem
    features = analisar_todos(imagem_pil)

    # Organizar os dados para predição
    X = np.array([[features["shi_tomasi"], features["harris"], features["contornos_verdes"]]])

    # Fazer predição
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]  # probabilidade de ser soja

    return {
        "possui_soja": bool(pred),
        "probabilidade_soja": round(prob * 100, 2),
        "shi_tomasi": features["shi_tomasi"],
        "harris": features["harris"],
        "contornos_verdes": features["contornos_verdes"]
    }
