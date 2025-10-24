import logging, asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np, cv2
from system.services.yolo_service import yolo_service

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)  # ajuste se tiver GPU boa

def _decode_and_infer(jpeg_bytes: bytes):
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    # chama YOLO em resolução menor (ganho gigante de FPS)
    results = yolo_service.model(frame, verbose=False, imgsz=640)
    annotated = results[0].plot()
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None

class WebSocketController:
    async def handle_websocket_connection(self, websocket: WebSocket):
        await websocket.accept()
        if not yolo_service.is_model_loaded():
            await websocket.send_text("ERRO: modelo indisponível")
            await websocket.close(); return
        try:
            loop = asyncio.get_running_loop()
            while True:
                data = await websocket.receive_bytes()
                # processa fora do loop
                out = await loop.run_in_executor(_executor, _decode_and_infer, data)
                if out is None:
                    continue
                # se cliente estiver lento, dropa em vez de travar
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_bytes(out)
        except WebSocketDisconnect:
            logger.info("WS desconectado")
        except Exception as e:
            logger.exception("WS erro: %s", e)
            try: await websocket.close()
            except: pass

websocket_controller = WebSocketController()