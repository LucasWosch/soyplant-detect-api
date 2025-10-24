from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

import system.auth as auth
from system.models.database_models import Utilizador
from system.models.pydantic_models import AnaliseResponse, HistoricoAnaliseResponse
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
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    local_texto: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session),
    current_user: Utilizador = Depends(auth.get_utilizador_atual)
):
    return await video_controller.analyze_video(file, latitude, longitude, local_texto, db, current_user)

@router.get("/historico", response_model=List[HistoricoAnaliseResponse])
async def get_historico_do_utilizador(
    db: AsyncSession = Depends(get_db_session),
    current_user: Utilizador = Depends(auth.get_utilizador_atual)
):
    return await video_controller.get_user_history(db, current_user)