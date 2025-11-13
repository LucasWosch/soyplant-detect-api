from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

import system.auth as auth
from system.models.database_models import Utilizador

from system.models.pydantic_models import (
    AnaliseResponse,
    HistoricoAnaliseResponse,
    VideoAnalysisRequest
)
from system.controllers.video_controller import video_controller
from system.database import get_db_session

router = APIRouter(
    prefix="/api/v1",
    tags=["Análise de Vídeo e Histórico"],
    # Aplica a dependência de autenticação a TODAS as rotas deste ficheiro
    dependencies=[Depends(auth.get_utilizador_atual)]
)


@router.post("/analisar-video", response_model=AnaliseResponse)
async def analisar_video(
        file: UploadFile = File(...),
        request_data: VideoAnalysisRequest = Depends(),
        db: AsyncSession = Depends(get_db_session)

):
    # Obtenha o usuário diretamente da função de autenticação
    current_user = auth.get_utilizador_atual()

    return await video_controller.analyze_video(file, request_data, db, current_user)

    return await video_controller.analyze_video(file, request_data, db, auth.get_utilizador_atual())


@router.get("/historico", response_model=List[HistoricoAnaliseResponse])
async def get_historico_do_utilizador(
        db: AsyncSession = Depends(get_db_session)

):
    # Obtenha o usuário diretamente da função de autenticação
    current_user = auth.get_utilizador_atual()

    return await video_controller.get_user_history(db, current_user)