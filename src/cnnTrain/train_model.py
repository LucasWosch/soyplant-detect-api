import os
import numpy as np
import pandas as pd
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Caminhos
dataset_path = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/src/dataset_soja_features.csv"  # CSV contendo colunas: arquivo, soja
imagens_path = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/all_images_v2"  # pasta com imagens reais

# Validar CSV
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

# Carregar CSV
df = pd.read_csv(dataset_path)
if not all(col in df.columns for col in ["arquivo", "soja"]):
    raise ValueError("CSV deve conter colunas 'arquivo' e 'soja'")

# Criar diretório temporário organizado por classes para ImageDataGenerator
temp_dir = "temp_dataset"
os.makedirs(os.path.join(temp_dir, "soja"), exist_ok=True)
os.makedirs(os.path.join(temp_dir, "nao_soja"), exist_ok=True)

for _, row in df.iterrows():
    origem = os.path.join(imagens_path, row["arquivo"])
    destino_dir = "soja" if row["soja"] == 1 else "nao_soja"
    destino = os.path.join(temp_dir, destino_dir, row["arquivo"])
    if os.path.exists(origem):
        with open(origem, 'rb') as fsrc:
            with open(destino, 'wb') as fdst:
                fdst.write(fsrc.read())

# Gerador de imagens com data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    temp_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    temp_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Definir a CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Classificação binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(train_gen, epochs=10, validation_data=val_gen)

print(history.history['accuracy'])
print(history.history['val_accuracy'])

# Salvar modelo
model.save("modelo_cnn_soja.h5")
print("Modelo salvo como modelo_cnn_soja.h5")
