import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("../dataset_soja_features.csv")
df = df[df["soja"].isin([0, 1])]  # filtra apenas os válidos

X = df[["shi_tomasi", "harris", "contornos_verdes"]]
y = df["soja"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(modelo, "modelo_soja_classificacao.pkl")
