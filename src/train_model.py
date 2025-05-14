import os
import cv2
import numpy as np
from keras import layers, models

# Caminhos ajustados em relação à localização do script
current_dir = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(current_dir, '../data/train')
val_folder = os.path.join(current_dir, '../data/val')

# Função para carregar imagens e rótulos
def load_data(data_folder, image_size=(128, 128)):
    images = []
    labels = []

    for image_name in os.listdir(data_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(data_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            image = image / 255.0
            images.append(image)

            # Rótulo binário fictício baseado no nome (ajuste conforme sua necessidade real)
            label = 0 if 'classe0' in image_name else 1
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Carregando os dados de treino e validação
X_train, y_train = load_data(train_folder)
X_val, y_val = load_data(val_folder)

# Adicionando canal único para imagens em escala de cinza
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Construção da CNN
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 128, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Treinamento do modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Salvando o modelo treinado
model.save(os.path.join(current_dir, '../soyplant_cnn_model.h5'))
