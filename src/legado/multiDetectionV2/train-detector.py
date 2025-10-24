# train_soja_multibox_v2.py
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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
N_BOXES = 10                   # número máx. de caixas por imagem
DATA_CSV = 'annotations.csv'  # CSV no formato: filename,class,xmin,ymin,xmax,ymax
IMAGES_DIR = '../../../data/v4/train/'
MODEL_SAVE_PATH = 'soja_detector_multibox_v2.keras'
SEED = 42

# filtros simples para qualidade das boxes (em proporção da imagem)
MIN_BOX_W = 1e-3
MIN_BOX_H = 1e-3

np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU: memory growth (evita OOM por pré-alocação)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Warn set_memory_growth:", e)

# =====================
# Leitura do CSV (formato V2)
# =====================
df = pd.read_csv(DATA_CSV, header=None)
# Garanta exatamente 6 colunas
if df.shape[1] != 6:
    raise ValueError(
        f"CSV deve ter 6 colunas (filename,class,xmin,ymin,xmax,ymax). Encontrado: {df.shape[1]}"
    )

df.columns = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

# (opcional) Se 'class' vier como string, tente converter
try:
    df['class'] = df['class'].astype(float)
except Exception:
    # se não der, mapeie para números arbitrários
    classes = {c: i for i, c in enumerate(sorted(df['class'].unique()))}
    df['class'] = df['class'].map(classes).astype(float)

# =====================
# Utilitários
# =====================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def xyxy_to_xywh_center_norm(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Converte (xmin,ymin,xmax,ymax) em (cx,cy,w,h), todos NORMALIZADOS em [0,1].
    """
    # corrige possíveis inversões/valores fora da imagem
    xmin = clamp(float(xmin), 0.0, img_w - 1.0)
    ymin = clamp(float(ymin), 0.0, img_h - 1.0)
    xmax = clamp(float(xmax), 0.0, img_w - 1.0)
    ymax = clamp(float(ymax), 0.0, img_h - 1.0)

    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2.0
    cy = ymin + h / 2.0

    # normaliza
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w / img_w
    h_n = h / img_h

    return cx_n, cy_n, w_n, h_n

def pad_or_truncate(boxes_xywh_conf, n_boxes=N_BOXES):
    """
    boxes_xywh_conf: lista de [conf, x, y, w, h] (normalizados).
    - Se houver > n_boxes, corta.
    - Se houver < n_boxes, preenche com zeros.
    Retorna (n_boxes, 5)
    """
    arr = np.zeros((n_boxes, 5), dtype=np.float32)
    m = min(len(boxes_xywh_conf), n_boxes)
    if m > 0:
        arr[:m] = np.array(boxes_xywh_conf[:m], dtype=np.float32)
    return arr

def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None, None, None
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    return img_resized, w, h

# =====================
# Agrupar anotações por imagem e converter para o target do modelo
# =====================
grouped = defaultdict(list)  # filename -> lista de linhas do CSV

for _, row in df.iterrows():
    grouped[row['filename']].append(row)

file_list = list(grouped.keys())

images = []
labels = []  # (N_BOXES, 5) -> [conf, x, y, w, h] normalizados

for filename in file_list:
    path = os.path.join(IMAGES_DIR, filename)
    img, img_w, img_h = load_image_rgb(path)

    # EXIBIR O IMG

    if img is None:
        # imagem não encontrada/corrompida
        continue

    boxes_this_image = []
    rows = grouped[filename]

    for r in rows:
        cx_n, cy_n, w_n, h_n = xyxy_to_xywh_center_norm(
            r['xmin'], r['ymin'], r['xmax'], r['ymax'], img_w, img_h
        )

        # descarta boxes muito pequenas (evita ruído/padding desnecessário)
        if w_n < MIN_BOX_W or h_n < MIN_BOX_H:
            continue

        # conf = 1.0 para boxes anotadas
        boxes_this_image.append([1.0, cx_n, cy_n, w_n, h_n])

    # garante shape (N_BOXES, 5)
    y = pad_or_truncate(boxes_this_image, N_BOXES)

    images.append(img)
    labels.append(y)

X = np.array(images, dtype=np.float32)
Y = np.array(labels, dtype=np.float32)

if len(X) == 0:
    raise RuntimeError("Nenhuma imagem válida encontrada após ler CSV e imagens. Verifique caminhos e CSV.")

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=SEED
)

# =====================
# Modelo
# =====================
def build_model(img_size=IMG_SIZE, n_boxes=N_BOXES):
    """
    Saída: (n_boxes, 5) com sigmoid -> [conf, x, y, w, h] normalizados [0..1]
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(n_boxes * 5, activation='sigmoid')(x)
    outputs = layers.Reshape((n_boxes, 5))(x)

    model = models.Model(inputs, outputs, name='soja_multibox_v2')
    return model

# =====================
# Loss: BCE para conf + L1 para bbox (mascarado por conf_true)
# =====================
def multibox_loss(y_true, y_pred):
    """
    y_*: (batch, N_BOXES, 5) -> [conf, x, y, w, h] em [0..1]
    - conf: Binary Cross Entropy
    - bbox: L1 (MAE) apenas onde conf_true == 1
    """
    conf_true = y_true[..., 0]
    conf_pred = y_pred[..., 0]
    bbox_true = y_true[..., 1:]
    bbox_pred = y_pred[..., 1:]

    # BCE por box e média no batch
    bce = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)
    bce = tf.reduce_mean(bce)

    # máscara (B, N, 1)
    mask = tf.cast(conf_true > 0.5, tf.float32)[..., tf.newaxis]
    l1 = tf.abs(bbox_true - bbox_pred) * mask  # (B, N, 4)
    # soma no N e coords; normaliza pela qtde de coords válidos
    valid = tf.reduce_sum(mask) * 4.0 + 1e-7
    l1 = tf.reduce_sum(l1) / valid

    return bce + l1

# =====================
# Treino
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
