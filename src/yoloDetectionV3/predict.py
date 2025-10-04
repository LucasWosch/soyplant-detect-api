import time
from typing import Tuple

import cv2
import numpy as np
from sort import Sort  # garanta que sort.py esteja acessível
import helpers as hp


# =========================
# CONFIGURAÇÕES
# =========================
MODEL_PATH: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/runs/detect/train6/weights/best.pt"
VIDEO_PATH: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/teste9.mp4"

OUTPUT_PATH_MP4: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.mp4"
OUTPUT_PATH_AVI: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.avi"

WINDOW_NAME: str = "Detecção - Soyplant"

CONF_THRES: float = 0.20
IOU_THRES: float = 0.45

# classe para o rótulo pré-tracking (opcional, já que seu modelo pode ter 1 classe principal)
DETECTION_LABEL_NAME: str = "soyplant"  # ajuste se quiser


def main() -> None:
    # — modelo / device —
    model, class_names = hp.load_model(MODEL_PATH)
    device, use_half = hp.device_and_half()

    # — vídeo —
    cap = hp.open_capture(VIDEO_PATH)
    first_frame = hp.read_first_frame(cap)
    src_fps = hp.get_source_fps(cap)
    target_h, target_w = first_frame.shape[:2]
    frame_size: Tuple[int, int] = (target_w, target_h)

    # — janela —
    use_display = hp.maybe_make_window(WINDOW_NAME)

    # — writer —
    writer = hp.try_create_writer(OUTPUT_PATH_MP4, frame_size, src_fps)
    out_path_final = OUTPUT_PATH_MP4
    if writer is None:
        writer = hp.try_create_writer(OUTPUT_PATH_AVI, frame_size, src_fps)
        out_path_final = OUTPUT_PATH_AVI
    if writer is None:
        cap.release()
        raise RuntimeError("Falha ao criar VideoWriter (mp4/avi).")

    # — tracker (SORT) —
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # — FPS —
    prev_time = time.time()
    smoothed_fps = None

    # ========= Primeiro frame =========
    detections0 = hp.yolo_detect(
        model=model,
        img=first_frame,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        device=device,
        use_half=use_half
    )

    img0 = first_frame.copy()

    # tracking do primeiro frame
    tracks0 = tracker.update(detections0)  # (M,5): [x1, y1, x2, y2, id]
    hp.draw_tracks(img0, tracks0, color=(255, 200, 0))

    # FPS
    now = time.time()
    inst_fps = 1.0 / max(1e-6, (now - prev_time))
    prev_time = now
    smoothed_fps = inst_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst_fps
    hp.put_fps(img0, smoothed_fps, org=(10, 30))

    # garante size e grava
    img0 = hp.resize_if_needed(img0, frame_size)
    writer.write(img0)
    if use_display:
        cv2.imshow(WINDOW_NAME, img0)

    # ========= Loop =========
    frame_count = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # detecção
            detections = hp.yolo_detect(
                model=model,
                img=frame,
                conf_thres=CONF_THRES,
                iou_thres=IOU_THRES,
                device=device,
                use_half=use_half
            )

            # desenho (pré-tracking)
            img = frame.copy()

            # tracking
            tracks = tracker.update(detections)  # (M,5): [x1, y1, x2, y2, id]
            hp.draw_tracks(img, tracks, color=(255, 200, 0))

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            smoothed_fps = 0.9 * (smoothed_fps if smoothed_fps else inst_fps) + 0.1 * inst_fps
            hp.put_fps(img, smoothed_fps, org=(10, 30))

            # saída
            img = hp.resize_if_needed(img, frame_size)
            writer.write(img)

            if use_display:
                cv2.imshow(WINDOW_NAME, img)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if frame_count % int(src_fps) == 0:
                print(f"[INFO] {frame_count} frames processados...")
            frame_count += 1

    finally:
        cap.release()
        try:
            writer.release()
        except Exception:
            pass
        if use_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("[OK] Finalizado. Arquivo salvo em:")
    if out_path_final and len(out_path_final) > 0:
        print("   -", out_path_final)


if __name__ == "__main__":
    main()
