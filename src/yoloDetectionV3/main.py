# main.py
# ============================================
# FastAPI para ingestão em tempo real (WebSocket / Webhook)
# Processamento com YOLO + SORT usando helpers.py e sort.py
# Grava vídeo de saída e (opcional) retransmite eventos por WS
# ============================================

import os
import io
import cv2
import time
import json
import base64
import threading
from queue import Queue, Empty
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body, Query, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sort import Sort  # garanta que sort.py esteja acessível
import helpers as hp    # garanta que helpers.py esteja acessível

# =========================
# CONFIGURAÇÕES DO MODELO
# =========================
MODEL_PATH: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/runs/detect/train8/weights/best.pt"
CONF_THRES: float = 0.20
IOU_THRES: float = 0.45
DETECTION_LABEL_NAME: str = "soyplant"

# =========================
# CONFIGURAÇÕES DE SAÍDA
# =========================
OUTPUT_PATH_MP4: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.mp4"
OUTPUT_PATH_AVI: str = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/saida_detectada.avi"
WINDOW_NAME: str = "Detecção - Soyplant (Server)"

# Quando a origem é streaming, definimos um FPS alvo para escrita
TARGET_OUTPUT_FPS: float = 30.0

# =========================
# TRACKER (SORT)
# =========================
SORT_MAX_AGE: int = 20
SORT_MIN_HITS: int = 3
SORT_IOU_THRESHOLD: float = 0.3

# =========================
# INGRESSO E EVENTOS
# =========================
# Tamanho máximo da fila de frames (para não estourar memória)
FRAME_QUEUE_MAXSIZE: int = 256

# Enviar evento a cada N frames (1 = todos)
SEND_EVERY_N_FRAMES: int = 1

# Se quiser retransmitir eventos para assinantes:
BROADCAST_EVENTS: bool = True  # eventos JSON em /ws/events

# =========================
# ESTADO GLOBAL DO SERVIDOR
# =========================
app = FastAPI(title="Soyplant Realtime Inference API", version="1.0.0")

# CORS opcional (ajuste conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fila de frames para processamento
frame_queue: "Queue[np.ndarray]" = Queue(maxsize=FRAME_QUEUE_MAXSIZE)

# Conexões que **enviam** frames (ingestão)
ingest_ws_clients: Set[WebSocket] = set()

# Conexões que **recebem** eventos (consumo)
event_ws_clients: Set[WebSocket] = set()

# Sinalização/recursos do pipeline
_processing_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_writer: Optional[cv2.VideoWriter] = None
_out_path_final: Optional[str] = None

# Modelo/Device/Tracker/FPS state
_model = None
_device = None
_use_half = False
_tracker: Optional[Sort] = None
_prev_time: Optional[float] = None
_smoothed_fps: Optional[float] = None
_frame_index: int = -1
_frame_size: Optional[Tuple[int, int]] = None
_video_meta: Dict[str, Any] = {
    "source": "realtime",
    "width": None,
    "height": None,
    "fps": TARGET_OUTPUT_FPS,
}


# =========================
# UTILITÁRIOS
# =========================
def _jpeg_to_ndarray(data: bytes) -> Optional[np.ndarray]:
    # Tenta decodificar bytes JPEG/PNG em ndarray BGR
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _broadcast_event(payload: Dict[str, Any]) -> None:
    if not BROADCAST_EVENTS:
        return
    if not event_ws_clients:
        return
    msg = json.dumps(payload, ensure_ascii=False)
    dead: List[WebSocket] = []
    for ws in list(event_ws_clients):
        try:
            # Starlette WebSocket é assíncrono; enviar via thread unsafe
            # então acumulamos para enviar no loop do evento? Para simplificar,
            # usamos try/except e .send_text via background com .run_until_complete
            # Aqui vamos usar um truque de "thread-safe" com .send_text via loop interno
            import anyio
            anyio.from_thread.run(ws.send_text, msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        try:
            event_ws_clients.discard(d)
            import anyio
            anyio.from_thread.run(d.close)
        except Exception:
            pass


def _make_event(frame_idx: int, fps: float, tracks: np.ndarray) -> Dict[str, Any]:
    # tracks esperado: (M,5) -> [x1,y1,x2,y2,id]
    tlist: List[Dict[str, Any]] = []
    if tracks is not None and len(tracks) > 0:
        for row in tracks:
            try:
                x1, y1, x2, y2, tid = row.tolist()
            except Exception:
                x1, y1, x2, y2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            tlist.append(
                {
                    "id": int(tid),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

    event = {
        "type": "soyplant-tracks",
        "ts_unix": time.time(),
        "frame_index": frame_idx,
        "fps": float(fps),
        "video_meta": dict(_video_meta),
        "tracks": tlist,
        "tag": DETECTION_LABEL_NAME,
    }
    return event


def _ensure_writer(frame_w: int, frame_h: int) -> None:
    global _writer, _out_path_final
    if _writer is not None:
        return
    _video_meta["width"] = frame_w
    _video_meta["height"] = frame_h
    _video_meta["fps"] = TARGET_OUTPUT_FPS

    writer = hp.try_create_writer(OUTPUT_PATH_MP4, (frame_w, frame_h), TARGET_OUTPUT_FPS)
    out_path = OUTPUT_PATH_MP4
    if writer is None:
        writer = hp.try_create_writer(OUTPUT_PATH_AVI, (frame_w, frame_h), TARGET_OUTPUT_FPS)
        out_path = OUTPUT_PATH_AVI
    if writer is None:
        raise RuntimeError("Falha ao criar VideoWriter (mp4/avi).")

    _writer = writer
    _out_path_final = out_path


# =========================
# WORKER DE PROCESSAMENTO
# =========================
def _processing_loop() -> None:
    global _prev_time, _smoothed_fps, _frame_index, _frame_size

    # Carrega modelo e inicializa device/half
    model, class_names = hp.load_model(MODEL_PATH)
    device, use_half = hp.device_and_half()

    # Tracker
    tracker = Sort(
        max_age=SORT_MAX_AGE,
        min_hits=SORT_MIN_HITS,
        iou_threshold=SORT_IOU_THRESHOLD
    )

    # Estado
    _prev_time = time.time()
    _smoothed_fps = None
    _frame_index = -1
    _frame_size = None

    # Loop
    while not _stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        if frame is None:
            continue

        # Inicializa tamanho/VideoWriter na 1ª imagem
        h, w = frame.shape[:2]
        if _frame_size is None:
            _frame_size = (w, h)
            _ensure_writer(w, h)
            try:
                hp.maybe_make_window(WINDOW_NAME)
            except Exception:
                pass

        # Detecção
        detections = hp.yolo_detect(
            model=model,
            img=frame,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            device=device,
            use_half=use_half
        )

        # Tracking
        tracks = tracker.update(detections)

        # FPS
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - _prev_time))
        _prev_time = now
        _smoothed_fps = inst_fps if _smoothed_fps is None else 0.9 * _smoothed_fps + 0.1 * inst_fps

        # Desenho
        img = frame.copy()
        hp.draw_tracks(img, tracks, color=(255, 200, 0))
        hp.put_fps(img, _smoothed_fps, org=(10, 30))
        img = hp.resize_if_needed(img, _frame_size)

        # Escrita
        if _writer is not None:
            _writer.write(img)

        # Exibição opcional (se houver backend)
        try:
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(1)
        except Exception:
            pass

        # Eventos
        _frame_index += 1
        if SEND_EVERY_N_FRAMES == 1 or (_frame_index % SEND_EVERY_N_FRAMES == 0):
            event = _make_event(_frame_index, _smoothed_fps, tracks)
            _broadcast_event(event)

    # Flush final
    try:
        if _writer is not None:
            _writer.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# =========================
# LIFECYCLE FASTAPI
# =========================
@app.on_event("startup")
def on_startup() -> None:
    global _processing_thread, _stop_event
    _stop_event.clear()
    t = threading.Thread(target=_processing_loop, name="processing-loop", daemon=True)
    t.start()
    _processing_thread = t


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    # força o loop a sair
    try:
        frame_queue.put_nowait(np.zeros((1, 1, 3), dtype=np.uint8))
    except Exception:
        pass
    if _processing_thread and _processing_thread.is_alive():
        _processing_thread.join(timeout=2.0)


# =========================
# ENDPOINTS
# =========================

@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "Soyplant Realtime Inference API"

# --- Ingestão via Webhook (HTTP POST) ---
# Aceita:
# 1) multipart/form-data com "file" (UploadFile)
# 2) body bruto (bytes da imagem)
# 3) JSON {"image_base64": "..."} (PNG/JPEG em base64)
@app.post("/ingest/frame")
async def ingest_frame(
    file: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Body(default=None),
    content_type: Optional[str] = Query(default=None, description="Override do Content-Type (image/jpeg, image/png).")
):
    try:
        if file is not None:
            data = await file.read()
        elif image_base64 is not None:
            data = base64.b64decode(image_base64)
        else:
            # Tenta ler o corpo cru (por exemplo, image/jpeg diretamente)
            # Starlette mantém o corpo no escopo da request; FastAPI abstrai
            # Precisamos de uma Request, mas para manter simples, rejeitamos aqui:
            return JSONResponse({"ok": False, "error": "Envie multipart 'file' ou JSON 'image_base64'."}, status_code=400)

        img = _jpeg_to_ndarray(data)
        if img is None:
            return JSONResponse({"ok": False, "error": "Falha ao decodificar imagem."}, status_code=400)

        try:
            frame_queue.put_nowait(img)
        except Exception:
            return JSONResponse({"ok": False, "error": "Fila cheia."}, status_code=429)

        return JSONResponse({"ok": True, "queued": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# --- Ingestão via WebSocket ---
# Cliente envia binário (JPEG/PNG) por mensagem.
@app.websocket("/ingest/ws")
async def ingest_ws(ws: WebSocket):
    await ws.accept()
    ingest_ws_clients.add(ws)
    try:
        while True:
            data = await ws.receive_bytes()
            img = _jpeg_to_ndarray(data)
            if img is None:
                # opcional: ignorar silenciosamente
                continue
            try:
                frame_queue.put_nowait(img)
            except Exception:
                # se fila cheia, descarta frame
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            ingest_ws_clients.discard(ws)
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

# --- WebSocket de eventos (broadcast) ---
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    event_ws_clients.add(ws)
    try:
        while True:
            # Mantemos conexão viva; cliente não precisa enviar nada
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            event_ws_clients.discard(ws)
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

# --- Status / caminho do arquivo de saída ---
@app.get("/status")
def status():
    return {
        "ok": True,
        "queue_size": frame_queue.qsize(),
        "frame_index": _frame_index,
        "smoothed_fps": _smoothed_fps,
        "output_path": _out_path_final,
        "video_meta": _video_meta,
    }


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    # Rode:  python main.py
    # Ou:    uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
