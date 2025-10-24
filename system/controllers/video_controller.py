import os
import uuid
from datetime import datetime
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging

from system.models.database_models import Utilizador, Analise
from system.services.video_processing import process_video_file
from system.services.yolo_service import yolo_service
# ESTE IMPORT AGORA FUNCIONA CORRETAMENTE
from system.utils.helpers import get_current_timestamp_iso

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoController:
    async def analyze_video(
            self,
            file: UploadFile,
            latitude: float,
            longitude: float,
            local_texto: str,
            db: AsyncSession,
            current_user: Utilizador
    ):
        if not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Tipo de ficheiro não suportado. Por favor, envie um vídeo.",
            )

        try:
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            save_path = os.path.join("processed_videos", unique_filename)

            os.makedirs("processed_videos", exist_ok=True)

            contagem = await process_video_file(file, save_path)

            nova_analise = Analise(
                utilizador_id=current_user.id,
                nome_arquivo_original=file.filename,
                video_salvo_em=save_path,
                data_analise=datetime.utcnow(),
                contagem_total_unicos=contagem,
                latitude=latitude,
                longitude=longitude,
                local_texto=local_texto
            )
            db.add(nova_analise)
            await db.commit()
            await db.refresh(nova_analise)

            logger.info(f"Análise {nova_analise.id} concluída para o utilizador {current_user.username}.")

            return {
                "id": nova_analise.id,
                "message": "Vídeo processado com sucesso!",
                "contagem_total_unicos": contagem
            }

        except Exception as e:
            logger.error(f"Erro ao processar o vídeo: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Não foi possível processar o vídeo. Erro: {str(e)}"
            )

    async def get_user_history(self, db: AsyncSession, current_user: Utilizador):
        query = select(Analise).where(Analise.utilizador_id == current_user.id).order_by(Analise.data_analise.desc())
        result = await db.execute(query)
        historico = result.scalars().all()
        return historico

    async def health_check(self):
        model_loaded = yolo_service.is_model_loaded()
        model_info = yolo_service.get_model_info() if model_loaded else None

        return {
            "status": "operacional" if model_loaded else "degradado",
            "timestamp": get_current_timestamp_iso(),
            "model_loaded": model_loaded,
            "model_info": model_info
        }


video_controller = VideoController()