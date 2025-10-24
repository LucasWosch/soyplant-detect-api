# Ficheiro: src/yoloDetectionV3/main.py

import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configura o path para importações corretas dentro do projeto
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Importações da Aplicação
from system.database import engine, Base
from system.controllers.video_controller import video_controller
from system.controllers.websocket_controller import websocket_controller
from system.routes import auth_routes, video_routes, public_routes

# Inicialização da API
app = FastAPI(
    title="Plataforma de Análise de Soja",
    version="3.2.0 (CORS Corrigido)",
    description="API para processar vídeos de plantações de soja, gerir utilizadores e consultar históricos."
)

# =====================================================================
# ✅ CORREÇÃO: Middleware de CORS
# =====================================================================
# Define explicitamente as origens que podem aceder à sua API.
# Incluímos ambos, localhost e 127.0.0.1, para resolver o erro.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # aceita qualquer origem (inclui Origin: null)
    allow_credentials=False,    # obrigatório se usar "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================

# Eventos de Startup
@app.on_event("startup")
async def on_startup():
    print("[INFO] Verificando e criando tabelas do banco de dados...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ [SUCESSO] Tabelas do banco de dados prontas.")

    print("\n🔍 VERIFICAÇÃO DO MODELO YOLO:")
    health_status = await video_controller.health_check()

    if health_status.get('model_loaded'):
        print("🎉 Modelo YOLO carregado com sucesso!")
        if health_status.get('model_info'):
            print(f"   📁 Local: {health_status['model_info'].get('path', 'N/A')}")
    else:
        print("❌ ATENÇÃO: Modelo YOLO não foi carregado!")

    print("=" * 60)
    print("🚀 API de Análise de Soja está online e pronta para receber requisições.")
    print("=" * 60)


# Inclusão dos Routers da API
app.include_router(public_routes.router)
app.include_router(auth_routes.router)
app.include_router(video_routes.router)


# Endpoint WebSocket
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_controller.handle_websocket_connection(websocket)


# Manuseamento de Erros Globais
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Ocorreu um erro interno no servidor."}
    )


# Ponto de Entrada (para desenvolvimento)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)