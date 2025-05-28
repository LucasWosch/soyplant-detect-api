import os
import cv2
import numpy as np
import pandas as pd
from keras import layers, models

# Caminhos ajustados
current_dir = os.path.dirname(os.path.abspath(__file__))
train_folder = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/sobel/train"
val_folder = "C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/DATASET/sobel/valid"

# CSV com informações de rótulo
csv_path = os.path.join(current_dir, '../../data/Imagens.csv')
df_labels = pd.read_csv(csv_path, sep=';')

# Função para carregar imagens e rótulos com base na planilha
def load_data(data_folder, label_column='soja', image_size=(128, 128)):
    images = []
    labels = []

    for image_name in os.listdir(data_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(data_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            image = image / 255.0
            images.append(image)

            # Buscar o rótulo no CSV
            row = df_labels[df_labels['nome_arquivo'] == image_name]
            if not row.empty:
                label = row.iloc[0][label_column]
                labels.append(label)
            else:
                print(f"Aviso: {image_name} não encontrado no CSV.")

    images = np.array(images)
    labels = np.array(labels).astype(np.float32)
    return images, labels

# Carregar dados
X_train, y_train = load_data(train_folder, label_column='soja')
X_val, y_val = load_data(val_folder, label_column='soja')

# Adicionar canal de cor
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Modelo cnnTrain atualizado
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),  # substitui InputLayer
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binário: soja/não soja
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinamento
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Salvar modelo
model.save(os.path.join(current_dir, 'soyplant_cnn_model.h5'))
