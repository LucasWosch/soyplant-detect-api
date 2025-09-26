import base64
import contextlib
import io
import math
import os
import time
from typing import Tuple, Optional

import cv2
import cvzone
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# ========= CONFIG =========
MODEL_PATH = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/runs/detect/train11/weights/best.pt"  # ajuste se necessário
VIDEO_PATH = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/teste14.mp4"
OUTPUT_PATH_MP4 = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.mp4"
OUTPUT_PATH_AVI = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.avi"
WINDOW_NAME = "Detecção - Soyplant"

# Parâmetros de inferência
CONF_THRES = 0.25
IOU_THRES = 0.45

# Device
_HAS_CUDA = torch.cuda.is_available()
_DEVICE_AUTO = 0 if _HAS_CUDA else "cpu"
_HALF_AUTO = bool(_HAS_CUDA)

# ==========================================
# Helpers
# ==========================================
def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def try_create_writer(
    path: str,
    frame_size: Tuple[int, int],
    fps: float,
    try_codecs = ("mp4v", "avc1", "XVID", "MJPG")
) -> Optional[cv2.VideoWriter]:
    """Tenta múltiplos codecs até abrir um writer válido."""
    ensure_parent_dir(path)
    w, h = frame_size
    for c in try_codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer is not None and writer.isOpened():
            print(f"[OK] VideoWriter aberto: {path} | codec={c} | {w}x{h} @ {fps:.2f} FPS")
            return writer
        else:
            try:
                writer.release()
            except Exception:
                pass
            print(f"[WARN] Falhou abrir VideoWriter com codec={c} -> {path}")
    return None

def make_window_or_none(name: str) -> bool:
    """Tenta abrir janela; se falhar, retorna False (modo headless)."""
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        return True
    except cv2.error as e:
        print(f"[INFO] Sem backend de GUI: {e}")
        return False

# ==========================================
# Modelo
# ==========================================
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # id -> name

# ==========================================
# Vídeo de entrada
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {VIDEO_PATH}")

# Lê o primeiro frame para travar tamanho/FPS corretos
ret, first_frame = cap.read()
if not ret or first_frame is None:
    cap.release()
    raise RuntimeError("Não foi possível ler o primeiro frame do vídeo.")

# FPS do source (fallback 30.0)
src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps <= 1.0 or src_fps != src_fps:  # NaN check
    src_fps = 30.0

# (Opcional) reduzir resolução aqui, se quiser:
# target_w, target_h = 1280, 720
# first_frame = cv2.resize(first_frame, (target_w, target_h))

target_h, target_w = first_frame.shape[:2]
frame_size = (target_w, target_h)

# Decide se terá display (tenta abrir janela)
USE_DISPLAY = make_window_or_none(WINDOW_NAME)

# ==========================================
# VideoWriter (sempre cria; salvar é obrigatório)
# ==========================================
writer = try_create_writer(OUTPUT_PATH_MP4, frame_size, src_fps)
out_path_final = OUTPUT_PATH_MP4
if writer is None:
    writer = try_create_writer(OUTPUT_PATH_AVI, frame_size, src_fps)
    out_path_final = OUTPUT_PATH_AVI
if writer is None:
    cap.release()
    raise RuntimeError(
        "Falha ao criar VideoWriter (mp4 e avi). "
        "Tente instalar codecs/usar outro caminho/evitar caracteres especiais."
    )

# ==========================================
# Loop
# ==========================================
prev_time = time.time()
smoothed_fps = None

def process_and_draw(img):
    results = model(
        img,
        stream=False,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=_DEVICE_AUTO,
        half=_HALF_AUTO,
        verbose=False
    )
    r = results[0]
    boxes = r.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

            cvzone.cornerRect(img, (x1, y1, w, h), l=12, t=2, rt=2)
            label = f"{cls_name} {conf:.2f}"
            cvzone.putTextRect(
                img, label,
                (max(0, x1), max(35, y1)),
                scale=0.8, thickness=1, offset=3
            )
    return img

# Processa o primeiro frame (já lido)
img0 = process_and_draw(first_frame)

# overlay FPS
now = time.time()
inst_fps = 1.0 / max(1e-6, (now - prev_time))
prev_time = now
smoothed_fps = inst_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst_fps
cv2.putText(img0, f"FPS: {smoothed_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# garante size do writer
if (img0.shape[1], img0.shape[0]) != frame_size:
    img0 = cv2.resize(img0, frame_size)

# escreve e (se possível) exibe
writer.write(img0)
if USE_DISPLAY:
    cv2.imshow(WINDOW_NAME, img0)

frame_count = 1
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        img = process_and_draw(frame)

        # FPS overlay
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps
        cv2.putText(img, f"FPS: {smoothed_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # garante size do writer
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)

        # salva SEMPRE
        writer.write(img)

        # exibe SE possível
        if USE_DISPLAY:
            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if frame_count % int(src_fps) == 0:
            print(f"[INFO] Processados {frame_count} frames...")
        frame_count += 1
finally:
    cap.release()
    try:
        writer.release()
    except Exception:
        pass
    if USE_DISPLAY:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

print("[OK] Finalizado. Arquivo salvo em:")
if os.path.exists(out_path_final):
    print("   -", out_path_final)
