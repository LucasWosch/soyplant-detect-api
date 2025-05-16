import os
import pandas as pd
from PIL import Image
from ..full_analysis import analisar_todos

pasta_imagens = os.getenv('RAW_IMAGES_PATH')
dados = []

for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.endswith((".jpg", ".png")):
        imagem_path = os.path.join(pasta_imagens, nome_arquivo)
        imagem = Image.open(imagem_path).convert("RGB")

        features = analisar_todos(imagem)

        # EX: definir a label baseado no nome do arquivo
        label = 1 if "soja" in nome_arquivo.lower() else 0

        dados.append({
            "arquivo": nome_arquivo,
            "shi_tomasi": features["shi_tomasi"],
            "harris": features["harris"],
            "contornos_verdes": features["contornos_verdes"],
            "label": label
        })

# Salvar como CSV
df = pd.DataFrame(dados)
df.to_csv("dataset_soja.csv", index=False)
