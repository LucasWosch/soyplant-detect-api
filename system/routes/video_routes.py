# src/yoloDetectionV3/routes/video_routes.py (ou onde estiverem as rotas de análise/histórico)

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

import system.auth as auth
from system.models.database_models import Utilizador, Analise
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


# NOVA ROTA: devolver o vídeo processado dessa análise
@router.get("/historico/{analise_id}/video")
async def get_video_da_analise(
    analise_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: Utilizador = Depends(auth.get_utilizador_atual)
):
    analise = await db.get(Analise, analise_id)
    if not analise:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análise não encontrada."
        )

    # garante que o vídeo é do utilizador logado
    if analise.utilizador_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Não tem permissão para aceder a este vídeo."
        )

    if not analise.video_salvo_em:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Esta análise não possui vídeo associado."
        )

    video_path = analise.video_salvo_em
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ficheiro de vídeo não encontrado no servidor."
        )

    filename = os.path.basename(video_path)

    # se quiser, dá pra ajustar o media_type com base na extensão
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename
    )