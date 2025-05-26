import os
import pandas as pd

# Caminho para o CSV e a pasta com os arquivos
caminho_csv = 'C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/Imagens de Validação.csv'  # CSV com a coluna 'nome_arquivo'
pasta_alvo = 'C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/Imagens Verdadeiro Positivo/valid'

# Lê o CSV e valida o separador correto
try:
    df = pd.read_csv(caminho_csv, sep=';')  # Ajuste para sep=',' se seu CSV usar vírgula
except Exception as e:
    print(f'Erro ao ler CSV: {e}')
    exit()

# Verifica se a coluna esperada existe
if 'nome_arquivo' not in df.columns:
    print("Erro: A coluna 'nome_arquivo' não foi encontrada no CSV.")
    exit()

# Cria um conjunto com os nomes dos arquivos permitidos
arquivos_permitidos = set(df['nome_arquivo'].dropna().astype(str).str.strip())

# Lista arquivos existentes na pasta
for arquivo in os.listdir(pasta_alvo):
    caminho_arquivo = os.path.join(pasta_alvo, arquivo)

    # Verifica se é arquivo (ignora subpastas)
    if os.path.isfile(caminho_arquivo):
        if arquivo not in arquivos_permitidos:
            os.remove(caminho_arquivo)
            print(f'Arquivo removido: {arquivo}')
        else:
            print(f'Arquivo mantido: {arquivo}')

print("Processamento finalizado.")