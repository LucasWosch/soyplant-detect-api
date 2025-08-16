# main.py
from io import BytesIO
from typing import Any, Dict, List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from object_counter import contar_objetos_pil
from green_detector import detectar_objetos_verdes
from feature_detector import detectar_harris, detectar_shi_tomasi
from sklearnTrain.full_analysis import analisar_todos
from kerasTrain.predict import predict_image
from sklearnTrain.predict import prever_se_soja
from vggAnnotation.predict import detectar_soja_na_imagem
from multiDetectionV1.predict import detectar_soja_multibox
from cnnTrain.predict import prever_com_cnn, prever_quantidade_cnn

TAGS_METADATA = [
    {"name": "Predições Clássicas", "description": "Predição binária e contagem com diferentes modelos."},
    {"name": "Detecção de Soja", "description": "Detecção de pés de soja e caixas (bounding boxes)."},
    {"name": "Contagem / Cor", "description": "Contagem de objetos e detecção por cor (verde)."},
    {"name": "Detecção de Features", "description": "Detectores de cantos/pontos: Harris e Shi-Tomasi."},
    {"name": "Análises Completas", "description": "Rotas que executam pipelines mais amplos de análise."}
]

app = FastAPI(
    title="API de Detecção/Contagem de Soja e Objetos",
    description=(
        "Endpoints para predição binária, detecção de múltiplos objetos, "
        "contagem e análise de imagens. Envie uma imagem (campo `file`) em multipart/form-data."
    ),
    version="1.0.1",
    openapi_tags=TAGS_METADATA,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
        "displayRequestDuration": True
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ErrorResponse(BaseModel):
    error: str = Field(...)

class PredictResponse(BaseModel):
    label: str
    confidence_percent: float
    raw_prediction: float

class BoxesResponse(BaseModel):
    total_boxes_detectados: int
    boxes: List[Any]

class CountObjectsResponse(BaseModel):
    total_objetos_detectados: int

class CountGreenResponse(BaseModel):
    total_verde_detectado: int

class PointsResponse(BaseModel):
    pontos_detectados: int

class FeaturesResponse(BaseModel):
    pontos_detectados_harris: int
    pontos_detectados_tomasi: int

class GenericDictResponse(BaseModel):
    result: Dict[str, Any]

def _read_image_from_upload(file: UploadFile) -> Image.Image:
    image_bytes = file.file.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API online. Acesse /docs para o Swagger ou /playground para testar uploads."}

@app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
def playground():
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
        img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }
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
          <h3>Resposta JSON</h3>
          <pre id="output">Aguardando requisição...</pre>
        </div>

        <div class="card" style="flex:1; min-width:320px;">
          <h3>Preview da Imagem</h3>
          <img id="preview" alt="Sem imagem ainda" />
        </div>
      </div>

      <script>
        async function send() {
          const endpoint = document.getElementById('endpoint').value;
          const fileInput = document.getElementById('file');
          const output = document.getElementById('output');
          const preview = document.getElementById('preview');
          output.textContent = "Enviando...";
          preview.removeAttribute('src');
          preview.alt = "Sem imagem ainda";

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

            // se a API retornar image_base64, mostramos
            if (json.image_base64) {
              preview.src = json.image_base64;
              preview.alt = "Imagem anotada";
            } else {
              preview.removeAttribute('src');
              preview.alt = "Este endpoint não retornou imagem.";
            }
          } catch (e) {
            output.textContent = "Erro: " + e;
            preview.removeAttribute('src');
            preview.alt = "Erro ao carregar imagem.";
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
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict(file: UploadFile = File(...)):
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
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_soja(file: UploadFile = File(...)):
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
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_cnn(file: UploadFile = File(...)):
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
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def predict_qty_cnn(file: UploadFile = File(...)):
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
    summary="Detecta caixas (VGG Annotation)",
    response_model=BoxesResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_soja_boxes(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        boxes = detectar_soja_na_imagem(image)
        return {"total_boxes_detectados": len(boxes), "boxes": boxes}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-box/",
    tags=["Detecção de Soja"],
    summary="Atalho para VGG Annotation (formato genérico)",
    response_model=GenericDictResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_box_alias(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_soja_na_imagem(image)
        return {"result": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-multibox/",
    tags=["Detecção de Soja"],
    summary="Detecta múltiplas caixas (multi model) e retorna imagem anotada",
    responses={400: {"model": ErrorResponse}}
)
async def detect_multibox_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        # agora a função já retorna boxes, raw nomeado e image_base64
        resultado = detectar_soja_multibox(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ---------------- Contagem / Cor ----------------

@app.post(
    "/count-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos (pipeline clássico)",
    response_model=CountObjectsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def count_objects(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        total_objetos = contar_objetos_pil(image)
        return {"total_objetos_detectados": total_objetos}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/count-green-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos verdes",
    response_model=CountGreenResponse,
    responses={400: {"model": ErrorResponse}}
)
async def count_green(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        total = detectar_objetos_verdes(image)
        return {"total_verde_detectado": total}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ---------------- Detecção de Features ----------------

@app.post(
    "/detect-shi-tomasi/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Shi-Tomasi)",
    response_model=PointsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_shi_tomasi_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        total = detectar_shi_tomasi(image)
        return {"pontos_detectados": total}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-harris/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris)",
    response_model=PointsResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_harris_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        total = detectar_harris(image)
        return {"pontos_detectados": total}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-features/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris + Shi-Tomasi)",
    response_model=FeaturesResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_features(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        total_harris = detectar_harris(image)
        total_tomasi = detectar_shi_tomasi(image)
        return {"pontos_detectados_harris": total_harris, "pontos_detectados_tomasi": total_tomasi}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
