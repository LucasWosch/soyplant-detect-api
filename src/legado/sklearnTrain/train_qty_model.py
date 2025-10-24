import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Caminho do dataset com features
caminho_dataset = "dataset_soja_features.csv"

# Verificar se o arquivo existe
if not os.path.exists(caminho_dataset):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dataset}")

# Carregar dataset
df = pd.read_csv(caminho_dataset)

# Verificar colunas obrigatórias
colunas_esperadas = ["shi_tomasi", "harris", "contornos_verdes", "quantidade_soja"]
for coluna in colunas_esperadas:
    if coluna not in df.columns:
        raise ValueError(f"Coluna esperada não encontrada no dataset: {coluna}")

# Tratar valores inválidos na coluna 'quantidade_soja'
df = df[df["quantidade_soja"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
df["quantidade_soja"] = df["quantidade_soja"].astype(float)

# Definir variáveis de entrada e saída
X = df[["shi_tomasi", "harris", "contornos_verdes"]]
y = df["quantidade_soja"]

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de regressão
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Avaliação do Modelo de Regressão:")
print(f"MAE (Erro Absoluto Médio): {mae:.2f}")
print(f"MSE (Erro Quadrático Médio): {mse:.2f}")
print(f"R² (Coeficiente de Determinação): {r2:.4f}")

# Salvar modelo
joblib.dump(modelo, "modelo_soja_regressao.pkl")
print("Modelo salvo com sucesso: modelo_soja_regressao.pkl")
