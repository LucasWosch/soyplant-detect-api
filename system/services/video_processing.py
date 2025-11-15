# system/services/video_processing.py

import os
import cv2
import tempfile
from typing import Set

import numpy as np
from fastapi import UploadFile, HTTPException

from system.services.yolo_service import yolo_service
from system.utils.sort import Sort
from system.utils.vis import draw_tracks

# MoviePy para conversão do vídeo para formato compatível com browser
from moviepy.editor import VideoFileClip


def convert_to_browser_friendly_mp4(input_path: str, output_path: str) -> None:
    """
    Converte um vídeo qualquer em um MP4 compatível com navegadores
    (H.264 + AAC) usando MoviePy (que internamente usa imageio-ffmpeg).

    :param input_path: caminho do vídeo de entrada (gerado pelo OpenCV)
    :param output_path: caminho final do vídeo convertido
    """
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=500,
            detail=f"Vídeo temporário para conversão não encontrado: {input_path}"
        )

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    try:
        # MoviePy lê o vídeo de entrada
        clip = VideoFileClip(input_path)

        # Gera o MP4 final com H.264 + AAC
        # (threads/preset podem ser ajustados se quiser mais desempenho/qualidade)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            fps=clip.fps or 25
        )

        clip.close()
    except Exception as e:
        print("[convert_to_browser_friendly_mp4] Erro MoviePy:", e)
        raise HTTPException(
            status_code=500,
            detail="Falha ao converter vídeo para formato compatível com navegador."
        )


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
      - Grava o vídeo processado em ficheiro temporário (extensão .mp4)
      - Converte esse ficheiro para MP4 (H.264 + AAC) em `output_path`
      - Retorna a contagem de IDs únicos rastreados (SORT)

    :param file: UploadFile vindo do FastAPI
    :param output_path: caminho FINAL do vídeo processado (compatível com browser)
    :param imgsz: tamanho máximo do lado maior para inferência
    :param conf_thres: threshold de confiança do YOLO
    :param iou_thres: threshold de IoU do YOLO
    :return: número de IDs únicos (plantas únicas) rastreados
    """

    if not yolo_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Modelo YOLO não está carregado."
        )

    # Garante pasta de saída final
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Caminho TEMPORÁRIO RAW
    # -------------------------
    # Mantemos a extensão do output (se não houver, forçamos .mp4)
    base_out, ext_out = os.path.splitext(output_path)
    if ext_out == "":
        ext_out = ".mp4"

    # Ex.: /.../processed_videos/webrtc_123.mp4  -> /.../processed_videos/webrtc_123_raw.mp4
    temp_output_path = f"{base_out}_raw{ext_out}"

    print("[process_video_file] output_path final:", output_path)
    print("[process_video_file] temp_output_path:", temp_output_path)

    # 1) Grava o upload em ficheiro temporário (entrada do OpenCV)
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
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
            status_code=400,
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
            status_code=400,
            detail="Dimensões do vídeo inválidas."
        )

    # 3) Cria VideoWriter para o vídeo processado TEMPORÁRIO
    # Aqui ainda usamos mp4v porque o MoviePy/ffmpeg interno vai recodificar depois.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[process_video_file] VideoWriter não abriu.")
        print("[process_video_file] temp_output_path:", temp_output_path)
        print("[process_video_file] fps:", fps, "size:", (width, height), "fourcc: mp4v")
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Não foi possível criar o vídeo de saída temporário."
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

            # Escreve frame processado no vídeo de saída TEMPORÁRIO
            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        # remove o vídeo de upload original
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 5) Converte o vídeo TEMPORÁRIO para um MP4 compatível com browser em output_path
    try:
        convert_to_browser_friendly_mp4(temp_output_path, output_path)
    finally:
        # remove o temporário do OpenCV em qualquer caso
        try:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except Exception:
            pass

    # Retorna o número de IDs únicos (pés de soja únicos rastreados)
    return len(seen_ids)
