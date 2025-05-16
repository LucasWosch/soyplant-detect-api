import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Carregar dataset
df = pd.read_csv("dataset_soja.csv")
X = df[["shi_tomasi", "harris", "contornos_verdes"]]
y = df["label"]

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Avaliar
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(modelo, "modelo_soja.pkl")
