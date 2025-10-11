# main.py
# ============================================
# WebSocket único /ws:
#  - recebe: [8 bytes float64 LE ts_client] + [JPEG]
#  - processa: YOLO + SORT e (opcional) YOLO tracker nativo
#  - responde: JSON por frame com tracks_sort, tracks_yolo, frame_b64 e TOTAIS de IDs únicos
# Não abre janela. (Opcional) grava vídeo com as caixas do SORT.
# ============================================

import os
import cv2
import time
import json
import base64
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sort import Sort               # seu tracker SORT
import helpers as hp                # helpers.py (IO, yolo_detect, draw_tracks)

# ============ YOLO tracker nativo (opcional) ============
YOLO_ENABLE: bool = True              # defina False se não tiver botsort/bytetrack YAML
YOLO_TRACKER: str = "bytetrack.yaml"  # ou "botsort.yaml" (precisa existir)

# =========================
# CONFIG GERAL
# =========================
MODEL_PATH: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/runs/detect/train8/weights/best.pt"
CONF_THRES: float = 0.20
IOU_THRES: float = 0.45

# Envio e gravação
SAVE_VIDEO: bool = True
OUTPUT_DIR: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos"
TARGET_OUTPUT_FPS: float = 30.0

# Enviar o frame renderizado (JPEG) no JSON (base64)
SEND_RENDERED_BASE64: bool = True
JPEG_QUALITY_RENDER: int = 75  # 50–85 é um bom range
WS_MAX_SEND_B64_BYTES = 2_000_000  # ~2 MB de base64 no JSON (ajuste)

# DEBUG
DEBUG_FAKE_BOXES: bool = False
DEBUG_FAKE_BOX_SIZE: int = 120

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Soyplant WS API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# =========================
# MODELO GLOBAL (carregado 1x)
# =========================
_yolo_det = None            # para hp.yolo_detect (detector)
_device = None
_use_half = False

_yolo_track_model = None    # para YOLO tracker nativo
_tracker_yaml = None

def _jpeg_to_ndarray(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_jpeg_b64(img: np.ndarray, quality: int = 75) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Falha ao codificar JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _yolo_track_to_ndarray(results) -> np.ndarray:
    """
    Converte resultado de model.track(frame) em (M,5) [x1,y1,x2,y2,id]
    """
    if not results:
        return np.empty((0,5), dtype=float)
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0,5), dtype=float)
    xyxy = boxes.xyxy
    ids = boxes.id
    if ids is None:
        return np.empty((0,5), dtype=float)
    xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
    ids = ids.int().cpu().numpy().reshape(-1) if hasattr(ids, "cpu") else ids.reshape(-1)
    out = np.concatenate([xyxy[:, :4], ids[:, None]], axis=1)
    return out.astype(float)

def _tracker_param(s: str) -> str:
    t = s.strip().lower()
    if t in ("botsort", "botsort.yaml"):   return "botsort.yaml"
    if t in ("bytetrack", "bytetrack.yaml"): return "bytetrack.yaml"
    if t.endswith((".yaml",".yml")):       return s
    return s + ".yaml"

def _make_fake_box(w: int, h: int, tid: int = 999) -> np.ndarray:
    side = min(DEBUG_FAKE_BOX_SIZE, w//3, h//3)
    cx, cy = w//2, h//2
    x1 = max(0, cx - side//2)
    y1 = max(0, cy - side//2)
    x2 = min(w-1, x1 + side)
    y2 = min(h-1, y1 + side)
    return np.asarray([[float(x1), float(y1), float(x2), float(y2), float(tid)]], dtype=float)

def _unique_out_path(base_dir: str, ext: str = "mp4") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = os.path.join(base_dir, f"saida_detectada_{ts}.{ext}")
    hp.ensure_parent_dir(p)
    return p

@app.on_event("startup")
def _load_models_once() -> None:
    global _yolo_det, _device, _use_half, _yolo_track_model, _tracker_yaml

    # detector (hp.yolo_detect usa ultralytics YOLO por baixo)
    _yolo_det, _ = hp.load_model(MODEL_PATH)
    _device, _use_half = hp.device_and_half()

    # YOLO tracker nativo (opcional)
    if YOLO_ENABLE:
        try:
            from ultralytics import YOLO as _U
            _yolo_track_model = _U(MODEL_PATH)
            _tracker_yaml = _tracker_param(YOLO_TRACKER)
            print(f"[BOOT] YOLO tracker ON -> {_tracker_yaml}")
        except Exception as e:
            print("[WARN] YOLO tracker OFF:", e)
            _yolo_track_model = None
            _tracker_yaml = None
    else:
        print("[BOOT] YOLO tracker OFF (YOLO_ENABLE=False)")

# =========================
# STATUS
# =========================
@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "Soyplant WS API (single /ws: recv jpeg+ts, send JSON tracks + frame_b64 + totals)"

@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "device": _device,
        "half": _use_half,
        "yolo_tracker_enabled": bool(_yolo_track_model is not None and _tracker_yaml is not None),
        "yolo_tracker_cfg": _tracker_yaml,
        "save_video": SAVE_VIDEO,
        "target_output_fps": TARGET_OUTPUT_FPS,
        "send_rendered_base64": SEND_RENDERED_BASE64,
        "jpeg_quality_render": JPEG_QUALITY_RENDER,
        "debug_fake_boxes": DEBUG_FAKE_BOXES,
    }

# =========================
# WEBSOCKET ÚNICO
# =========================
@app.websocket("/ws")
async def ws_bidirectional(ws: WebSocket):
    """
    Protocolo:
      - Cliente envia: [8 bytes float64 LE ts_client] + [JPEG] (frame binário)
      - Servidor responde: JSON com:
          * tracks_sort, tracks_yolo
          * frame_b64 (render do servidor)
          * totals: contadores de IDs únicos vistos na sessão
    """
    await ws.accept()

    writer: Optional[cv2.VideoWriter] = None
    frame_size: Optional[Tuple[int, int]] = None
    frame_index: int = -1
    prev_time: Optional[float] = None
    smoothed_fps: Optional[float] = None

    # Tracker SORT por conexão
    tracker_sort = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # === ACUMULADORES DE IDS (totais da sessão) ===
    seen_sort_ids: set[int] = set()
    seen_yolo_ids: set[int] = set()

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                data = message["bytes"]
            elif "text" in message and message["text"] is not None:
                # ignore textos (pings/comandos futuros)
                continue
            else:
                continue

            if not data or len(data) <= 8:
                continue

            # 1) parse: timestamp + jpeg
            try:
                ts_client = np.frombuffer(memoryview(data)[:8], dtype="<f8", count=1)[0]
            except Exception as e:
                print("[WS] Erro ao ler timestamp:", e)
                continue

            jpeg = data[8:]
            frame = _jpeg_to_ndarray(jpeg)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if frame_size is None:
                frame_size = (w, h)
                if SAVE_VIDEO:
                    writer = hp.try_create_writer(_unique_out_path(OUTPUT_DIR, "mp4"), frame_size, TARGET_OUTPUT_FPS)
                    if writer is None:
                        writer = hp.try_create_writer(_unique_out_path(OUTPUT_DIR, "avi"), frame_size, TARGET_OUTPUT_FPS)

            # 2) detecção (para SORT)
            try:
                detections = hp.yolo_detect(
                    model=_yolo_det,
                    img=frame,
                    conf_thres=CONF_THRES,
                    iou_thres=IOU_THRES,
                    device=_device,
                    use_half=_use_half
                )
            except Exception as e:
                print("[WS] Erro em yolo_detect:", e)
                continue

            # 3) tracking SORT
            try:
                tracks_sort = tracker_sort.update(detections)
            except Exception as e:
                print("[WS] Erro no SORT:", e)
                tracks_sort = np.empty((0, 5), dtype=float)

            # 4) tracking YOLO (opcional)
            tracks_yolo = None
            if _yolo_track_model is not None and _tracker_yaml is not None:
                try:
                    results = _yolo_track_model.track(
                        source=frame,
                        persist=True,
                        tracker=_tracker_yaml,
                        conf=CONF_THRES,
                        iou=IOU_THRES,
                        device=_device,
                        half=_use_half,
                        verbose=False,
                    )
                    tracks_yolo = _yolo_track_to_ndarray(results)
                except Exception as e:
                    print("[WS] YOLO tracker erro:", e)
                    tracks_yolo = None

            # 5) debug: caixa fake
            if DEBUG_FAKE_BOXES and (len(tracks_sort) == 0) and (tracks_yolo is None or len(tracks_yolo) == 0):
                tracks_sort = _make_fake_box(w, h, tid=999)

            # === acumula IDs únicos ===
            if tracks_sort is not None and len(tracks_sort) > 0:
                for row in tracks_sort:
                    try:
                        seen_sort_ids.add(int(row[4]))
                    except Exception:
                        pass
            if tracks_yolo is not None and len(tracks_yolo) > 0:
                for row in tracks_yolo:
                    try:
                        seen_yolo_ids.add(int(row[4]))
                    except Exception:
                        pass

            # 6) FPS
            now = time.time()
            inst_fps = TARGET_OUTPUT_FPS if prev_time is None else 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            smoothed_fps = inst_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst_fps

            # 7) render (desenha ambos trackers)
            img_render = frame.copy()
            # SORT (amarelo claro)
            hp.draw_tracks(img_render, tracks_sort, color=(255, 200, 0))
            # YOLO (vermelho)
            if tracks_yolo is not None and len(tracks_yolo) > 0:
                hp.draw_tracks(img_render, tracks_yolo, color=(0, 0, 255))
            hp.put_fps(img_render, float(smoothed_fps), org=(10, 30))

            if SAVE_VIDEO and writer is not None:
                W, H = frame_size
                if (img_render.shape[1], img_render.shape[0]) != (W, H):
                    imgw = cv2.resize(img_render, (W, H))
                else:
                    imgw = img_render
                writer.write(imgw)

            # 8) payload com totais
            frame_index += 1
            payload = {
                "type": "soyplant-tracks",
                "ts_server": now,
                "ts_client": float(ts_client),
                "frame_index": frame_index,
                "fps": float(smoothed_fps) if smoothed_fps is not None else None,
                "video_meta": {"width": w, "height": h, "fps": TARGET_OUTPUT_FPS},
                "tracks_sort": [
                    {"id": int(t[4]), "x1": float(t[0]), "y1": float(t[1]), "x2": float(t[2]), "y2": float(t[3])}
                    for t in (tracks_sort if tracks_sort is not None else [])
                ],
                "tracks_yolo": [
                    {"id": int(t[4]), "x1": float(t[0]), "y1": float(t[1]), "x2": float(t[2]), "y2": float(t[3])}
                    for t in (tracks_yolo if tracks_yolo is not None else [])
                ],
                "totals": {
                    "sort_unique_ids": len(seen_sort_ids),
                    "yolo_unique_ids": len(seen_yolo_ids),
                }
            }

            if SEND_RENDERED_BASE64:
                try:
                    b64 = _encode_jpeg_b64(img_render, JPEG_QUALITY_RENDER)
                    if len(b64) <= WS_MAX_SEND_B64_BYTES:
                        payload["frame_b64"] = b64
                    else:
                        payload["frame_b64"] = None
                except Exception as e:
                    print("[WS] Falha ao gerar frame_b64:", e)
                    payload["frame_b64"] = None

            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                print("[WS] Erro ao enviar JSON:", e)
                break

    except WebSocketDisconnect:
        print("[WS] cliente desconectou")
    except Exception as e:
        import traceback
        print("[WS] ERRO fatal:\n", "".join(traceback.format_exception(e)))
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

# =========================
# Webhook (opcional — inalterado)
# =========================
@app.post("/ingest/frame")
async def ingest_frame(
    file: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Body(default=None),
    content_type: Optional[str] = Query(default=None),
):
    try:
        if file is not None:
            data = await file.read()
        elif image_base64 is not None:
            data = base64.b64decode(image_base64)
        else:
            return JSONResponse({"ok": False, "error": "Envie multipart 'file' ou JSON 'image_base64'."}, status_code=400)

        img = _jpeg_to_ndarray(data)
        if img is None:
            return JSONResponse({"ok": False, "error": "Falha ao decodificar imagem."}, status_code=400)
        return JSONResponse({"ok": True, "decoded": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
