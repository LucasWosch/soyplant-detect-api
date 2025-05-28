import os
import numpy as np
import pandas as pd
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from PIL import Image
from sklearn.model_selection import train_test_split

# Caminhos
csv_path = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/src/dataset_soja_features.csv"
image_dir = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/all_images"

# Verificações
if not os.path.exists(csv_path):
    raise FileNotFoundError("Arquivo CSV não encontrado.")

# Carregar dados
df = pd.read_csv(csv_path)
if not all(col in df.columns for col in ["arquivo", "quantidade_soja"]):
    raise ValueError("CSV deve conter as colunas 'arquivo' e 'quantidade_soja'")

# Remover registros inválidos em quantidade_soja
df = df[df["quantidade_soja"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

# Preparar X (imagens) e y (quantidade)
X = []
y = []

for _, row in df.iterrows():
    caminho_img = os.path.join(image_dir, row["arquivo"])
    if os.path.exists(caminho_img):
        img = Image.open(caminho_img).convert("RGB").resize((128, 128))
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(float(row["quantidade_soja"]))

X = np.array(X)
y = np.array(y)

# Dividir treino/validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo CNN para regressão
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # saída contínua para regressão
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Treinar modelo
model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=16)

# Salvar modelo
model.save("modelo_cnn_soja_regressor.h5")
print("Modelo salvo como modelo_cnn_soja_regressor.h5")
