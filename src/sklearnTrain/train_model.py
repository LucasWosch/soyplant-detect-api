import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Caminho do dataset com features
caminho_dataset = "dataset_soja_features.csv"

# Verificar se o arquivo existe
if not os.path.exists(caminho_dataset):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dataset}")

# Carregar dataset
df = pd.read_csv(caminho_dataset)

# Verificar colunas obrigatórias
colunas_esperadas = ["shi_tomasi", "harris", "contornos_verdes", "soja"]
for coluna in colunas_esperadas:
    if coluna not in df.columns:
        raise ValueError(f"Coluna esperada não encontrada no dataset: {coluna}")

# Definir variáveis de entrada e saída
X = df[["shi_tomasi", "harris", "contornos_verdes"]]
y = df["soja"]  # 0 ou 1

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar
y_pred = modelo.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(modelo, "modelo_soja_classificacao.pkl")
print("Modelo salvo com sucesso: modelo_soja_classificacao.pkl")
