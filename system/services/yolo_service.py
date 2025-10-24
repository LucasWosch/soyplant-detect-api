# Ficheiro: src/yoloDetectionV3/services/yolo_service.py

import logging
from ultralytics import YOLO
import cv2
from system.config.settings import MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloService:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model = self._load_model()

    def _load_model(self):
        try:
            logger.info(f"A carregar o modelo YOLO de: {self.model_path}")
            model = YOLO(self.model_path)
            logger.info("Modelo YOLO carregado com sucesso.")
            return model
        except Exception as e:
            logger.error(f"FALHA AO CARREGAR O MODELO YOLO: {e}")
            return None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> dict:
        if self.is_model_loaded():
            return {"path": self.model_path, "type": "YOLOv8"}
        return None

    def track_video(self, video_path: str) -> int:
        """
        Processa um vídeo para detetar e RASTREAR objetos únicos (plantas),
        retornando uma contagem precisa.
        """
        if not self.is_model_loaded():
            logger.error("Tentativa de processar vídeo sem modelo carregado.")
            return 0

        # =====================================================================
        # CORREÇÃO PRINCIPAL: Usar o método track() do YOLOv8
        # =====================================================================

        # O argumento 'persist=True' diz ao tracker para se lembrar dos objetos entre os frames.
        # 'conf=0.33' é o nosso limiar de confiança otimizado.
        # 'tracker="bytetrack.yaml"' especifica um dos melhores algoritmos de tracking.
        results_generator = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            conf=0.33,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_ids = set()

        # Itera sobre os resultados frame a frame
        for results in results_generator:
            # Verifica se existem caixas de tracking (boxes com IDs)
            if results.boxes.id is not None:
                # Converte os IDs para uma lista de inteiros
                ids = results.boxes.id.int().cpu().tolist()
                # Adiciona os IDs vistos neste frame ao nosso conjunto
                for obj_id in ids:
                    tracked_ids.add(obj_id)

        contagem_final = len(tracked_ids)
        logger.info(
            f"Análise de tracking concluída para '{video_path}'. Contagem final de plantas únicas: {contagem_final}")

        return contagem_final


yolo_service = YoloService()