import os
import pandas as pd

# Caminhos
caminho_csv = 'C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/Imagens.csv'  # Ex: 'dados/arquivos.csv'
pasta_arquivos = 'C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/all_images v2'  # Ex: 'dados/imagens'
caminho_saida = 'C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/ImagensV2.csv'  # Ex: 'dados/arquivo_filtrado.csv'

# Lê o CSV
df = pd.read_csv(caminho_csv, sep=';')

# Verificar se os arquivos existem na pasta
def arquivo_existe(nome_arquivo):
    caminho_completo = os.path.join(pasta_arquivos, nome_arquivo)
    return os.path.isfile(caminho_completo)

# Filtrar DataFrame
df_filtrado = df[df['nome_arquivo'].apply(arquivo_existe)]

# Salvar CSV resultante
df_filtrado.to_csv(caminho_saida, index=False)

print(f'Arquivo salvo com {len(df_filtrado)} linhas em: {caminho_saida}')
