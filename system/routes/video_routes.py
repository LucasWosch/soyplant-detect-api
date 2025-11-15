# src/yoloDetectionV3/routes/video_routes.py

import os
import mimetypes
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

import system.auth as auth
from system.models.database_models import Utilizador, Analise
from system.models.pydantic_models import AnaliseResponse, HistoricoAnaliseResponse
from system.controllers.video_controller import video_controller
from system.database import get_db_session

# Calcula diretório raiz do projeto (ajuste se sua estrutura for diferente)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
PROCESSED_VIDEOS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "system", "processed_videos"))

print("[video_routes] PROJECT_ROOT =", PROJECT_ROOT)
print("[video_routes] PROCESSED_VIDEOS_DIR =", PROCESSED_VIDEOS_DIR)

router = APIRouter(
    prefix="/api/v1",
    tags=["Análise de Vídeo e Histórico"],
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


@router.get("/historico/{analise_id}/video")
async def get_video_da_analise(
    analise_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: Utilizador = Depends(auth.get_utilizador_atual)
):
    # 1) Busca a análise
    analise = await db.get(Analise, analise_id)
    if not analise:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análise não encontrada."
        )

    # 2) Garante que é do usuário logado
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

    # 3) Monta o caminho do arquivo corretamente
    path_db = analise.video_salvo_em
    print(f"[get_video_da_analise] analise_id={analise_id}, video_salvo_em={path_db}")

    # Se for caminho absoluto, usa direto
    if os.path.isabs(path_db):
        video_path = path_db
    else:
        # Se vier algo tipo "system/processed_videos/arquivo.mp4"
        # junta com PROJECT_ROOT
        if path_db.startswith("system" + os.sep) or path_db.startswith("system/"):
            video_path = os.path.join(PROJECT_ROOT, path_db)
        # Se vier algo tipo "processed_videos/arquivo.mp4" ou só "arquivo.mp4"
        else:
            # Se o valor tiver subpasta, usa ele dentro de PROCESSED_VIDEOS_DIR
            # Ex.: "processed_videos/arquivo.mp4" -> pega só o nome do arquivo
            filename = os.path.basename(path_db)
            video_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)

    video_path = os.path.normpath(video_path)
    print(f"[get_video_da_analise] Caminho final do vídeo: {video_path}")

    if not os.path.exists(video_path):
        print(f"[get_video_da_analise] Arquivo NÃO encontrado: {video_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ficheiro de vídeo não encontrado no servidor."
        )

    # 4) Descobre content-type
    media_type, _ = mimetypes.guess_type(video_path)
    if not media_type:
        media_type = "video/mp4"

    filename = os.path.basename(video_path)

    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=filename
    )
