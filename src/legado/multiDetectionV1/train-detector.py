# train_soja_multibox.py
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from collections import defaultdict

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
N_BOXES = 2  # número máximo de caixas por imagem
DATA_CSV = 'annotations.csv'
IMAGES_DIR = '../../../data/v2/'
MODEL_SAVE_PATH = 'soja_detector_multibox_b2.keras'
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================
# Carrega CSV e agrupa anotações por imagem
# =====================
df = pd.read_csv(DATA_CSV)

# Garante colunas necessárias
required_cols = {'filename', 'class_id', 'x_center', 'y_center', 'width', 'height'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f'Faltam colunas no CSV: {missing}')

# Agrupa por arquivo para lidar com múltiplas caixas por imagem
grouped = defaultdict(list)
for _, row in df.iterrows():
    grouped[row['filename']].append([
        float(row['class_id']), float(row['x_center']), float(row['y_center']),
        float(row['width']), float(row['height'])
    ])

file_list = list(grouped.keys())

# =====================
# Funções utilitárias
# =====================
def pad_or_truncate(boxes, n_boxes=N_BOXES):
    """
    boxes: lista de [conf, x, y, w, h] normalizados (0..1).
    - Se houver > n_boxes, corta.
    - Se houver < n_boxes, completa com zeros (conf=0).
    Retorna (n_boxes, 5)
    """
    arr = np.zeros((n_boxes, 5), dtype=np.float32)
    m = min(len(boxes), n_boxes)
    if m > 0:
        arr[:m] = np.array(boxes[:m], dtype=np.float32)
    return arr

def load_image(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    return img

def build_model(img_size=IMG_SIZE, n_boxes=N_BOXES):
    """
    Saída: (n_boxes, 5) com sigmoid -> [conf, x, y, w, h] em [0..1]
    """


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
        layers.Dense(n_boxes * 5, activation='sigmoid'),  # [class_id, x, y, w, h]
        layers.Reshape((n_boxes, 5))
    ])
    return model

def iou_xywh(box1, box2, eps=1e-7):
    """
    box = [x, y, w, h] centro-normalizado [0..1].
    IOU em espaço normalizado (não é usado no loss base, mas pode ser útil).
    """
    # converte para xyxy
    def to_xyxy(b):
        x, y, w, h = b
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return x1, y1, x2, y2

    x1a, y1a, x2a, y2a = to_xyxy(box1)
    x1b, y1b, x2b, y2b = to_xyxy(box2)
    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = area_a + area_b - inter + eps
    return inter / union

# =====================
# Monta dataset em memória (simples e direto)
# =====================
images = []
labels = []  # shape (N_BOXES, 5)

for filename in file_list:
    path = os.path.join(IMAGES_DIR, filename)
    img = load_image(path)
    if img is None:
        continue
    boxes = grouped[filename]  # lista de [conf, x, y, w, h]
    y = pad_or_truncate(boxes, N_BOXES)
    images.append(img)
    labels.append(y)

X = np.array(images, dtype=np.float32)
Y = np.array(labels, dtype=np.float32)  # (num_imgs, N_BOXES, 5)

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=SEED
)

# =====================
# Loss: BCE para conf + L1 para bbox somente quando conf_true=1
# =====================
def multibox_loss(y_true, y_pred):
    """
    y_* shape: (batch, N_BOXES, 5) => [conf, x, y, w, h] em [0..1]
    - conf: Binary Cross Entropy
    - bbox: L1 (MAE) apenas onde conf_true == 1 (ignora caixas de padding)
    """
    conf_true = y_true[..., 0]
    conf_pred = y_pred[..., 0]
    bbox_true = y_true[..., 1:]
    bbox_pred = y_pred[..., 1:]

    bce = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)

    # Máscara das caixas válidas
    mask = tf.cast(tf.greater(conf_true, 0.5), tf.float32)  # (B, N)
    mask = tf.expand_dims(mask, axis=-1)                    # (B, N, 1)

    l1 = tf.abs(bbox_true - bbox_pred)                      # (B, N, 4)
    l1 = l1 * mask                                          # aplica máscara
    l1 = tf.reduce_sum(l1, axis=[1, 2]) / (tf.reduce_sum(mask) + 1e-7)

    # BCE médio por box
    bce = tf.reduce_mean(bce)

    # Peso relativo (ajuste se quiser)
    loss = bce + l1
    return loss

# =====================
# Modelo e treino
# =====================
model = build_model(IMG_SIZE, N_BOXES)
model.compile(optimizer='adam', loss=multibox_loss, metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")
