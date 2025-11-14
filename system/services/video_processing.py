# system/services/video_processing.py

import os
import cv2
import tempfile
from typing import Optional, Tuple, Set

import numpy as np
from fastapi import UploadFile, HTTPException, status

from system.services.yolo_service import yolo_service
from system.utils.sort import Sort
from system.utils.vis import draw_tracks


async def process_video_file(
    file: UploadFile,
    output_path: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> int:
    """
    Processa o vídeo enviado:
      - Salva upload em arquivo temporário
      - Aplica YOLO + SORT frame a frame
      - Desenha as detecções no frame
      - Grava o vídeo processado em `output_path`
      - Retorna a contagem de IDs únicos rastreados (SORT)

    :param file: UploadFile vindo do FastAPI
    :param output_path: caminho final do vídeo processado
    :param imgsz: tamanho máximo do lado maior para inferência
    :param conf_thres: threshold de confiança do YOLO
    :param iou_thres: threshold de IoU do YOLO
    :return: número de IDs únicos (plantas únicas) rastreados
    """

    if not yolo_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo YOLO não está carregado."
        )

    # Garante pasta de saída
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 1) Grava o upload em ficheiro temporário
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao salvar ficheiro temporário: {e}"
        )

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        # limpa o temporário se der erro
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível abrir o vídeo enviado."
        )

    # 2) Lê metadados do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 25.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dimensões do vídeo inválidas."
        )

    # 3) Cria VideoWriter para o vídeo processado
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # saída .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Não foi possível criar o vídeo de saída."
        )

    # 4) Tracker SORT e conjunto de IDs únicos
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    seen_ids: Set[int] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Redimensiona para inferência (mantendo proporção) ---
            h, w, _ = frame.shape
            scale = 1.0
            if max(h, w) > imgsz:
                scale = imgsz / max(h, w)
                img_infer = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA
                )
            else:
                img_infer = frame

            try:
                # YOLO (modelo Ultralytics)
                # Aqui assumindo que yolo_service.model é um objeto YOLO do ultralytics
                results = yolo_service.model(
                    img_infer,
                    conf=conf_thres,
                    iou=iou_thres,
                    verbose=False
                )

                boxes = results[0].boxes
                dets = np.empty((0, 5), dtype=np.float32)

                if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                    conf = (
                        boxes.conf.cpu().numpy().astype(np.float32)
                        if boxes.conf is not None
                        else np.ones((xyxy.shape[0],), dtype=np.float32)
                    )

                    # reescala para coordenadas do frame original
                    if scale != 1.0:
                        xyxy /= scale

                    dets = np.hstack([xyxy, conf.reshape(-1, 1)])

            except Exception as e:
                # em caso de erro num frame específico, apenas loga e segue
                print(f"[process_video_file] Erro no YOLO: {e}")
                dets = np.empty((0, 5), dtype=np.float32)

            # Atualiza tracker SORT
            try:
                tracks = tracker.update(dets)
            except Exception as e:
                print(f"[process_video_file] Erro no SORT: {e}")
                tracks = np.empty((0, 5), dtype=np.float32)

            # Regista IDs únicos
            if tracks is not None and len(tracks) > 0:
                for row in tracks:
                    try:
                        seen_ids.add(int(row[4]))
                    except Exception:
                        pass

            # Desenha as caixas no frame
            draw_tracks(frame, tracks, color=(255, 200, 0))

            # Escreve frame processado no vídeo de saída
            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Retorna o número de IDs únicos (pés de soja únicos rastreados)
    return len(seen_ids)
