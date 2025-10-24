# Ficheiro: src/yoloDetectionV3/services/video_processing.py

import logging
from fastapi import UploadFile, HTTPException, status

# ESTE IMPORT AGORA VAI FUNCIONAR CORRETAMENTE
from system.services.yolo_service import yolo_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_video_file(file: UploadFile, save_path: str) -> int:
    """
    Salva o ficheiro de vídeo enviado e depois o processa com o YOLO para contar objetos.
    """
    try:
        logger.info(f"A salvar o vídeo em: {save_path}")
        file_content = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info("Vídeo salvo com sucesso.")

        if not yolo_service.is_model_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo YOLO não está carregado. Não é possível processar o vídeo."
            )

        logger.info(f"Iniciando análise YOLO para o ficheiro: {save_path}")
        contagem = yolo_service.track_video(video_path=save_path)
        logger.info(f"Análise concluída. Contagem total: {contagem}")

        return contagem

    except Exception as e:
        logger.error(f"Falha crítica no processamento de vídeo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o ficheiro de vídeo: {str(e)}"
        )