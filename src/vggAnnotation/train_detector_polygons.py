import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 8
DATA_CSV = 'soyplant_v3.csv'
IMAGES_DIR = '../../data/v3/'
MODEL_SAVE_PATH = 'soja_segmentation_model.keras'

# Função para carregar os polígonos do CSV (VIA)
def carregar_anotacoes(csv_path):
    df = pd.read_csv(csv_path)
    anotacoes = []

    for _, row in df.iterrows():
        try:
            shape_data = json.loads(row['region_shape_attributes'])
            if shape_data.get('name') != 'polygon':
                continue
            x = shape_data['all_points_x']
            y = shape_data['all_points_y']
            pontos = list(zip(x, y))
            anotacoes.append({'filename': row['filename'], 'polygon': pontos})
        except:
            continue  # Pula linhas sem anotação válida

    return anotacoes

# Gera imagem e máscara binária com base nos polígonos
def carregar_dados(anotacoes, img_dir):
    imagens = []
    mascaras = []

    for anot in tqdm(anotacoes):
        img_path = os.path.join(img_dir, anot['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        escala_x = IMG_SIZE / w_orig
        escala_y = IMG_SIZE / h_orig

        # Escala o polígono
        pontos_redimensionados = np.array([
            [int(x * escala_x), int(y * escala_y)]
            for x, y in anot['polygon']
        ])

        # Cria máscara binária
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillPoly(mask, [pontos_redimensionados], color=1)

        imagens.append(img.astype('float32') / 255.0)
        mascaras.append(np.expand_dims(mask, axis=-1))  # (H, W, 1)

    return np.array(imagens), np.array(mascaras)

# Carrega e prepara os dados
anotacoes = carregar_anotacoes(DATA_CSV)
X, y = carregar_dados(anotacoes, IMAGES_DIR)

# Divide em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CNN para segmentação (simples estilo mini U-Net)
def build_segmentation_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    return model

# Compilação do modelo
model = build_segmentation_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=BATCH_SIZE)

# Salva o modelo
model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")
