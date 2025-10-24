# Ficheiro: src/yoloDetectionV3/controllers/websocket_controller.py

import logging
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import cv2

# CORREÇÃO: O nome da classe é YoloService (e não YOLOService)
# Também vamos importar a instância 'yolo_service' que já está carregada.
from system.services.yolo_service import yolo_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketController:
    async def handle_websocket_connection(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("Cliente WebSocket conectado.")

        if not yolo_service.is_model_loaded():
            logger.warning("Conexão WebSocket recebida, mas o modelo YOLO não está carregado.")
            await websocket.send_text("ERRO: Modelo de análise não está disponível.")
            await websocket.close()
            return

        try:
            while True:
                # Recebe os bytes da imagem do cliente
                bytes_data = await websocket.receive_bytes()

                # Converte os bytes para uma imagem que o OpenCV possa usar
                nparr = np.frombuffer(bytes_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Processa o frame com o modelo YOLO
                results = yolo_service.model(frame, verbose=False)

                # Anota o frame com as deteções
                annotated_frame = results[0].plot()

                # Codifica o frame anotado para enviar de volta ao cliente
                _, buffer = cv2.imencode('.jpg', annotated_frame)

                # Envia a imagem processada de volta
                await websocket.send_bytes(buffer.tobytes())

        except WebSocketDisconnect:
            logger.info("Cliente WebSocket desconectado.")
        except Exception as e:
            logger.error(f"Erro na conexão WebSocket: {e}")
            await websocket.close()


websocket_controller = WebSocketController()