import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse

from system.controllers.video_controller import video_controller

router = APIRouter(
    tags=["Status e Interfaces"]
)

@router.get("/", response_class=PlainTextResponse, include_in_schema=False)
def root():
    return "API do TCC está online. Acesse /docs para a documentação."

@router.get("/interface", response_class=FileResponse, include_in_schema=False)
def get_main_interface():
    path_to_html = "interface.html"
    if not os.path.exists(path_to_html):
        raise HTTPException(status_code=404, detail="Ficheiro da interface principal não encontrado.")
    return FileResponse(path_to_html)

@router.get("/health")
async def health_check():
    """Verifica a saúde do serviço, incluindo o status do modelo YOLO."""
    return await video_controller.health_check()

@router.get("/status")
async def status():
    """Fornece um status geral da aplicação."""
    health = await video_controller.health_check()
    return {
        "status": "online",
        "timestamp": health.get("timestamp"),
        "model_loaded": health.get("model_loaded"),
        "database": "PostgreSQL"
    }