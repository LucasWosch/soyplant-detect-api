import os
from typing import Tuple, Optional, List, Dict

import cv2
import cvzone
import numpy as np
import torch
from ultralytics import YOLO


# =========================
# Sistema de arquivos / I/O
# =========================
def ensure_parent_dir(path: str) -> None:
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
            print(f"[OK] VideoWriter: {path} | codec={c} | {w}x{h} @ {fps:.2f} FPS")
            return writer
        # tente liberar, se possível
        try:
            writer.release()
        except Exception:
            pass
        print(f"[WARN] Falha VideoWriter codec={c} -> {path}")
    return None


def open_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    return cap


def read_first_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Não foi possível ler o primeiro frame do vídeo.")
    return frame


def get_source_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps != fps:  # NaN check
        fps = 30.0
    return fps


def maybe_make_window(name: str) -> bool:
    """Tenta abrir janela; se falhar, retorna False (modo headless)."""
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        return True
    except cv2.error as e:
        print(f"[INFO] Sem backend de GUI: {e}")
        return False


def resize_if_needed(img, frame_size: Tuple[int, int]):
    h, w = img.shape[:2]
    if (w, h) != frame_size:
        return cv2.resize(img, frame_size)
    return img


# =========================
# Modelo / Inferência YOLO
# =========================
def load_model(model_path: str):
    model = YOLO(model_path)
    class_names = model.names  # id -> name
    return model, class_names


def device_and_half():
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    use_half = bool(has_cuda)
    return device, use_half


def yolo_detect(
    model: YOLO,
    img,
    conf_thres: float,
    iou_thres: float,
    device,
    use_half: bool
) -> np.ndarray:
    """
    Roda YOLO no frame e retorna detecções no formato (N,5): [x1, y1, x2, y2, conf].
    Não desenha nada aqui (puro cálculo).
    """
    results = model(
        img,
        stream=False,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        half=use_half,
        verbose=False
    )
    r = results[0]
    boxes = r.boxes

    dets: List[List[float]] = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            dets.append([float(int(x1)), float(int(y1)), float(int(x2)), float(int(y2)), conf])

    if len(dets) == 0:
        return np.empty((0, 5))
    return np.asarray(dets, dtype=float)

def draw_tracks(
    img,
    tracks: np.ndarray,
    color=(255, 200, 0)
) -> None:
    """
    Desenha caixas do tracker (SORT) no formato [x1, y1, x2, y2, id].
    """
    if tracks is None or len(tracks) == 0:
        return
    for x1, y1, x2, y2, tid in tracks.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID {int(tid)}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def put_fps(img, fps_value: float, org=(10, 30)) -> None:
    cv2.putText(img, f"FPS: {fps_value:.1f}", org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
