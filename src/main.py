# main.py
from io import BytesIO
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

# ==== IMPORTS DOS SEUS MÓDULOS (mantidos) ====
# import tensorflow as tf
from object_counter import contar_objetos_pil
from green_detector import detectar_objetos_verdes
from feature_detector import detectar_harris, detectar_shi_tomasi
from sklearnTrain.full_analysis import analisar_todos
from kerasTrain.predict import predict_image
from sklearnTrain.predict import prever_se_soja
from vggAnnotation.predict import detectar_soja_na_imagem
from multiDetectionV1.predict import detectar_soja_multibox
from cnnTrain.predict import prever_com_cnn, prever_quantidade_cnn
# ============================================

# ===================== OpenAPI / Swagger Metadata =====================
TAGS_METADATA = [
    {
        "name": "Predições Clássicas",
        "description": "Predição binária e contagem com diferentes modelos."
    },
    {
        "name": "Detecção de Soja",
        "description": "Detecção de pés de soja e caixas (bounding boxes)."
    },
    {
        "name": "Contagem / Cor",
        "description": "Contagem de objetos e detecção por cor (verde)."
    },
    {
        "name": "Detecção de Features",
        "description": "Detectores de cantos/pontos: Harris e Shi-Tomasi."
    },
    {
        "name": "Análises Completas",
        "description": "Rotas que executam pipelines mais amplos de análise."
    }
]

app = FastAPI(
    title="API de Detecção/Contagem de Soja e Objetos",
    description=(
        "Endpoints para predição binária, detecção de múltiplos objetos, "
        "contagem e análise de imagens. Envie uma **imagem** (campo `file`) "
        "em `multipart/form-data`."
    ),
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
        "displayRequestDuration": True
    }
)

# CORS (opcional, útil se for chamar do frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas (Pydantic) =====================
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Mensagem de erro.")

class PredictResponse(BaseModel):
    label: str = Field(..., example="Pé de soja detectado")
    confidence_percent: float = Field(..., example=97.53)
    raw_prediction: float = Field(..., example=0.9753)

class BoxesResponse(BaseModel):
    total_boxes_detectados: int = Field(..., example=3)
    # Estrutura dos boxes pode variar; mantemos livre:
    boxes: List[Any] = Field(..., example=[{"x": 12, "y": 34, "w": 100, "h": 80, "score": 0.91}])

class CountObjectsResponse(BaseModel):
    total_objetos_detectados: int = Field(..., example=42)

class CountGreenResponse(BaseModel):
    total_verde_detectado: int = Field(..., example=18)

class PointsResponse(BaseModel):
    pontos_detectados: int = Field(..., example=256)

class FeaturesResponse(BaseModel):
    pontos_detectados_harris: int = Field(..., example=120)
    pontos_detectados_tomasi: int = Field(..., example=98)

# Saídas “livres” (o seu código retorna dict dinâmico):
class GenericDictResponse(BaseModel):
    result: Dict[str, Any] = Field(..., description="Objeto de resultado dinâmico.")


# ===================== Helpers =====================
def _read_image_from_upload(file: UploadFile) -> Image.Image:
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


# ===================== Rotas =====================

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API online. Acesse /docs para o Swagger ou /playground para testar uploads."}


@app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
def playground():
    """
    Página simples de testes com input de imagem, select de endpoint e preview da resposta.
    """
    html = """
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Playground - Upload de Imagem</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; }
        label { display:block; margin-top:12px; }
        button { margin-top:16px; padding:8px 14px; cursor:pointer; }
        pre { background:#111; color:#eee; padding:12px; border-radius:8px; overflow-x:auto; }
        .row { display:flex; gap:24px; flex-wrap: wrap; }
        .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; }
      </style>
    </head>
    <body>
      <h1>Playground de Upload</h1>
      <p>Selecione um endpoint, escolha uma imagem e clique em Enviar.</p>
      <div class="row">
        <div class="card">
          <label for="endpoint">Endpoint</label>
          <select id="endpoint">
            <option value="/predict/">POST /predict/</option>
            <option value="/detect-soja-boxes/">POST /detect-soja-boxes/</option>
            <option value="/predict-soja/">POST /predict-soja/</option>
            <option value="/predict-cnn/">POST /predict-cnn/</option>
            <option value="/predict-qty-cnn/">POST /predict-qty-cnn/</option>
            <option value="/count-objects/">POST /count-objects/</option>
            <option value="/count-green-objects/">POST /count-green-objects/</option>
            <option value="/detect-shi-tomasi/">POST /detect-shi-tomasi/</option>
            <option value="/detect-harris/">POST /detect-harris/</option>
            <option value="/detect-features/">POST /detect-features/</option>
            <option value="/analyze-all/">POST /analyze-all/</option>
            <option value="/detect-box/">POST /detect-box/</option>
            <option value="/detect-multibox/">POST /detect-multibox/</option>
          </select>

          <label for="file">Imagem</label>
          <input type="file" id="file" accept="image/*"/>

          <button onclick="send()">Enviar</button>
        </div>

        <div class="card" style="flex:1; min-width:320px;">
          <h3>Resposta</h3>
          <pre id="output">Aguardando requisição...</pre>
        </div>
      </div>

      <script>
        async function send() {
          const endpoint = document.getElementById('endpoint').value;
          const fileInput = document.getElementById('file');
          const output = document.getElementById('output');
          output.textContent = "Enviando...";

          if (!fileInput.files.length) {
            output.textContent = "Selecione uma imagem primeiro.";
            return;
          }

          const formData = new FormData();
          formData.append('file', fileInput.files[0]);

          try {
            const resp = await fetch(endpoint, {
              method: 'POST',
              body: formData
            });
            const json = await resp.json();
            output.textContent = JSON.stringify(json, null, 2);
          } catch (e) {
            output.textContent = "Erro: " + e;
          }
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ---------------- Predições Clássicas ----------------

@app.post(
    "/predict/",
    tags=["Predições Clássicas"],
    summary="Predição binária (se é pé de soja)",
    description="Recebe uma imagem e retorna rótulo, confiança (%) e probabilidade bruta.",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        prediction = predict_image(image)
        confidence = round(float(prediction) * 100, 2)
        label = "Pé de soja detectado" if prediction > 0.5 else "Nenhum pé de soja detectado"
        return PredictResponse(
            label=label,
            confidence_percent=confidence,
            raw_prediction=round(float(prediction), 4)
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/predict-soja/",
    tags=["Predições Clássicas"],
    summary="Predição (modelo sklearn) se é soja",
    description="Executa `prever_se_soja` e retorna o dicionário completo do modelo.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_soja(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = prever_se_soja(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/predict-cnn/",
    tags=["Predições Clássicas"],
    summary="Predição (modelo CNN)",
    description="Executa `prever_com_cnn` e retorna o dicionário completo.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_cnn(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = prever_com_cnn(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/predict-qty-cnn/",
    tags=["Predições Clássicas"],
    summary="Predição de quantidade (CNN)",
    description="Executa `prever_quantidade_cnn` e retorna o dicionário completo.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_qty_cnn(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = prever_quantidade_cnn(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------- Detecção de Soja ----------------

@app.post(
    "/detect-soja-boxes/",
    tags=["Detecção de Soja"],
    summary="Detecta caixas (bounding boxes) de soja (VGG Annotation)",
    description="Executa `detectar_soja_na_imagem` e retorna a quantidade e as boxes.",
    response_model=BoxesResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_soja_boxes(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        boxes = detectar_soja_na_imagem(image)
        return BoxesResponse(total_boxes_detectados=len(boxes), boxes=boxes)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/detect-box/",
    tags=["Detecção de Soja"],
    summary="Detecta caixas (atalho para VGG Annotation)",
    description="Atalho mantendo sua rota original `/detect-box/`. Retorna o dicionário/objeto do seu método.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_box_alias(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_soja_na_imagem(image)
        # Alguns projetos retornam lista; normalizamos como 'result'
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/detect-multibox/",
    tags=["Detecção de Soja"],
    summary="Detecta múltiplas caixas (multi model)",
    description="Executa `detectar_soja_multibox` e retorna o dicionário/objeto completo.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_multibox(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_soja_multibox(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------- Contagem / Cor ----------------

@app.post(
    "/count-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos (pipeline clássico)",
    description="Executa `contar_objetos_pil` e retorna o total detectado.",
    response_model=CountObjectsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def count_objects(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        total_objetos = contar_objetos_pil(image)
        return CountObjectsResponse(total_objetos_detectados=total_objetos)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/count-green-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos verdes",
    description="Executa `detectar_objetos_verdes` e retorna o total detectado.",
    response_model=CountGreenResponse,
    responses={400: {"model": ErrorResponse}}
)
async def count_green(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        total = detectar_objetos_verdes(image)
        return CountGreenResponse(total_verde_detectado=total)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------- Detecção de Features ----------------

@app.post(
    "/detect-shi-tomasi/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Shi-Tomasi)",
    description="Executa `detectar_shi_tomasi` e retorna a contagem.",
    response_model=PointsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_shi_tomasi_route(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        total = detectar_shi_tomasi(image)
        return PointsResponse(pontos_detectados=total)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/detect-harris/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris)",
    description="Executa `detectar_harris` e retorna a contagem.",
    response_model=PointsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_harris_route(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        total = detectar_harris(image)
        return PointsResponse(pontos_detectados=total)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/detect-features/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris + Shi-Tomasi)",
    description="Executa ambos detectores e retorna as contagens.",
    response_model=FeaturesResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_features(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        total_harris = detectar_harris(image)
        total_tomasi = detectar_shi_tomasi(image)
        return FeaturesResponse(
            pontos_detectados_harris=total_harris,
            pontos_detectados_tomasi=total_tomasi
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------- Análise Completa ----------------

@app.post(
    "/analyze-all/",
    tags=["Análises Completas"],
    summary="Executa pipeline completo de análise",
    description="Executa `analisar_todos` e retorna o dicionário/objeto completo.",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def analyze_all(file: UploadFile = File(..., description="Imagem (multipart/form-data)")):
    try:
        image = _read_image_from_upload(file)
        resultado = analisar_todos(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ===================== Execução local =====================
if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
