import os
import pandas as pd
from PIL import Image
from full_analysis import analisar_todos  # ajuste conforme sua estrutura de pastas

# Caminho das imagens e da planilha
current_dir = os.path.dirname(os.path.abspath(__file__))
pasta_imagens = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/all_images"

csv_path = os.path.join(current_dir, '../../data/Imagens.csv')
df_labels = pd.read_csv(csv_path, sep=';')

dados = []

for _, row in df_labels.iterrows():
    nome_arquivo = row["nome_arquivo"]
    label_binaria = row["soja"]
    quantidade_soja = row["quantidade_soja"]

    imagem_path = os.path.join(pasta_imagens, nome_arquivo)
    if not os.path.exists(imagem_path):
        print(f"Imagem não encontrada: {imagem_path}")
        continue

    imagem = Image.open(imagem_path).convert("RGB")
    features = analisar_todos(imagem)

    dados.append({
        "arquivo": nome_arquivo,
        "shi_tomasi": features["shi_tomasi"],
        "harris": features["harris"],
        "contornos_verdes": features["contornos_verdes"],
        "soja": label_binaria,
        "quantidade_soja": quantidade_soja
    })

# Salvar novo dataset com features
df = pd.DataFrame(dados)
df.to_csv("dataset_soja_features.csv", index=False)
