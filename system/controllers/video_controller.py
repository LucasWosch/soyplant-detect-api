import os
import uuid
from datetime import datetime
from fastapi import UploadFile, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging

# Importações dos modelos e exceções customizadas
from system.models.database_models import Utilizador, Analise
from system.models.pydantic_models import VideoAnalysisRequest
from system.services.video_processing import process_video_file
from system.services.yolo_service import yolo_service
from system.exceptions import ValidationError, FileProcessingError
from system.utils.helpers import get_current_timestamp_iso

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoController:
    async def analyze_video(
        self,
        file: UploadFile,
        request_data: VideoAnalysisRequest,
        db: AsyncSession,
        current_user: Utilizador
    ):
        """
        Analisa um vídeo de soja, validando o arquivo e os dados de entrada,
        processando com YOLO e salvando o resultado no banco de dados.
        """
        # --- Validação do Arquivo ---
        MAX_FILE_SIZE_MB = 500
        ALLOWED_MIME_TYPES = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv']

        # Lê o conteúdo do arquivo para validar o tamanho
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValidationError(detail=f"Arquivo muito grande. Tamanho máximo permitido: {MAX_FILE_SIZE_MB}MB.")

        if file.content_type not in ALLOWED_MIME_TYPES:
            raise ValidationError(
                detail=f"Tipo de arquivo não suportado: {file.content_type}. Tipos permitidos: {', '.join(ALLOWED_MIME_TYPES)}.")

        # Se as validações passaram, processamos o vídeo
        try:
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            save_path = os.path.join("processed_videos", unique_filename)
            os.makedirs("processed_videos", exist_ok=True)

            with open(save_path, "wb") as buffer:
                buffer.write(file_content)

            logger.info(f"Vídeo salvo em: {save_path}")

            # Chama o serviço de processamento
            # Nota: A função process_video_file pode precisar de um ajuste para receber o conteúdo em memória
            # em vez do objeto UploadFile, para evitar ler o arquivo duas vezes.
            # Por ora, mantemos a lógica original.
            contagem = await process_video_file(file, save_path)

            nova_analise = Analise(
                utilizador_id=current_user.id,
                nome_arquivo_original=file.filename,
                video_salvo_em=save_path,
                data_analise=datetime.utcnow(),
                contagem_total_unicos=contagem,
                latitude=request_data.latitude,  # Usa os dados validados do Pydantic
                longitude=request_data.longitude,  # Usa os dados validados do Pydantic
                local_texto=request_data.local_texto  # Usa os dados validados do Pydantic
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
            # Usa nossa exceção customizada para erros de processamento
            raise FileProcessingError(detail=f"Não foi possível processar o vídeo. Erro: {str(e)}")

    async def get_user_history(self, db: AsyncSession, current_user: Utilizador):
        """Retorna o histórico de análises para o utilizador atual."""
        query = select(Analise).where(Analise.utilizador_id == current_user.id).order_by(Analise.data_analise.desc())
        result = await db.execute(query)
        historico = result.scalars().all()
        return historico

    async def health_check(self):
        """Verifica a saúde do serviço, incluindo o status do modelo YOLO."""
        model_loaded = yolo_service.is_model_loaded()
        model_info = yolo_service.get_model_info() if model_loaded else None

        return {
            "status": "operacional" if model_loaded else "degradado",
            "timestamp": get_current_timestamp_iso(),
            "model_loaded": model_loaded,
            "model_info": model_info
        }


video_controller = VideoController()