import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 8
DATA_CSV = 'annotations.csv'
IMAGES_DIR = '../../data/v2/'
MODEL_SAVE_PATH = 'soja_detector_model.keras'

# Carrega CSV
df = pd.read_csv(DATA_CSV)

# Converte para arrays (imagens e rótulos)
def load_data(df):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0

        label = np.array([row['class_id'], row['x_center'], row['y_center'], row['width'], row['height']], dtype='float32')

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

X, y = load_data(df)

# Divide em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CNN simples para detectar bounding box
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='sigmoid')  # [class_id, x, y, w, h]
    ])
    return model

def custom_loss(y_true, y_pred):
    class_loss = tf.keras.losses.BinaryCrossentropy()(y_true[:, 0], y_pred[:, 0])
    bbox_loss = tf.reduce_mean(tf.square(y_true[:, 1:] - y_pred[:, 1:]))
    return class_loss + bbox_loss

model = build_model()
model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])

# Treina o modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=BATCH_SIZE)

# Salva o modelo
model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")
