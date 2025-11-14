# main.py
from io import BytesIO
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from object_counter import contar_objetos_pil
from green_detector import detectar_objetos_verdes
from feature_detector import detectar_harris, detectar_shi_tomasi
from src.legado.sklearnTrain.full_analysis import analisar_todos
from src.legado.kerasTrain.predict import predict_image
from src.legado.sklearnTrain.predict import prever_se_soja
from src.legado.vggAnnotation.predict import detectar_soja_na_imagem
from src.legado.multiDetectionV1.predict import detectar_soja_multibox
from src.legado.multiDetectionV2.predict import detectar_soja_multibox_V2
from src.legado.cnnTrain.predict import prever_com_cnn, prever_quantidade_cnn

# --- YOLO (CPU) ---
from src.legado.yoloDetectionV1.predict import predict_yolo  # garante que MODEL_PATH esteja correto no yolo_predict.py
from src.legado.yoloDetectionV2.predict import predict_yolo_V2  # garante que MODEL_PATH esteja correto no yolo_predict.py

TAGS_METADATA = [
    {"name": "Predições Clássicas", "description": "Predição binária e contagem com diferentes modelos."},
    {"name": "Detecção de Soja", "description": "Detecção de pés de soja e caixas (bounding boxes)."},
    {"name": "Detecção de Soja V2", "description": "Versão v2 multi-box normalizada [conf,x,y,w,h]."},
    {"name": "YOLO (CPU)", "description": "Detecção com Ultralytics YOLOv8 rodando no CPU."},
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
    version="1.1.0",
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

# ===================== Models (Swagger) =====================
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

# YOLO response models (opcional – melhora docs)
class YOLOBoxPixel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    name: str

class YOLOBoxNorm(BaseModel):
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    name: str

class YOLOResponse(BaseModel):
    boxes: List[YOLOBoxPixel]
    boxes_norm: List[YOLOBoxNorm]
    raw: List[Dict[str, Any]]
    image_base64: Optional[str] = None
    arquitetura: Optional[str] = None   # resumo textual do model.info()
    model: Optional[str] = None         # string com \n
    model_lines: Optional[List[str]] = None  # <<< mesma info, linha a linha

# ===================== Utils =====================
def _read_image_from_upload(file: UploadFile) -> Image.Image:
    image_bytes = file.file.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")

# ===================== Root & Playground =====================
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
        select, input[type=file] { width: 100%; }
        #gallery { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap:16px; margin-top:8px; }
        figure { margin:0; }
        figcaption { font-size:12px; margin-top:6px; word-break: break-all; color:#666; }
        .muted { color:#777; font-size: 12px; }
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
            <option value="/detect-multibox-V2/">POST /detect-multibox-V2/</option>
            <option value="/detect-yolo/">POST /detect-yolo/</option>
            <option value="/detect-yolo-v2/">POST /detect-yolo-v2/</option>
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
          <h3>Preview da Imagem (image_base64)</h3>
          <img id="preview" alt="Sem imagem ainda" />
          <h4 style="margin-top:16px;">Todas as imagens retornadas</h4>
          <div id="gallery"><p class="muted">Sem imagens ainda.</p></div>
        </div>
      </div>

      <javascript>
        function isDataUrlImage(value) {
          return typeof value === 'string' && value.startsWith('data:image/');
        }

        // percorre recursivamente o JSON e coleta todas as chaves com dataURL
        function collectImages(obj, pathPrefix = '') {
          const items = [];
          if (obj && typeof obj === 'object') {
            for (const [k, v] of Object.entries(obj)) {
              const path = pathPrefix ? pathPrefix + '.' + k : k;
              if (isDataUrlImage(v)) {
                items.push({ key: path, dataUrl: v });
              } else if (v && typeof v === 'object') {
                items.push(...collectImages(v, path));
              }
            }
          }
          return items;
        }

        // ordena naturalmente se a chave começa com número (ex: "1_original")
        function naturalSortImages(items) {
          const numFromKey = (k) => {
            const last = k.split('.').pop() || '';
            const m = /^\\s*(\\d+)/.exec(last);
            return m ? parseInt(m[1], 10) : Number.POSITIVE_INFINITY;
          };
          return items.sort((a, b) => {
            const na = numFromKey(a.key), nb = numFromKey(b.key);
            if (na !== nb) return na - nb;
            return a.key.localeCompare(b.key);
          });
        }

        function renderGallery(images) {
          const gallery = document.getElementById('gallery');
          gallery.innerHTML = '';
          if (!images.length) {
            gallery.innerHTML = '<p class="muted">Este endpoint não retornou imagens.</p>';
            return;
          }
          for (const item of images) {
            const fig = document.createElement('figure');
            const img = document.createElement('img');
            const cap = document.createElement('figcaption');
            img.src = item.dataUrl;
            img.alt = item.key;
            cap.textContent = item.key;
            fig.appendChild(img);
            fig.appendChild(cap);
            gallery.appendChild(fig);
          }
        }

        async function send() {
          const endpoint = document.getElementById('endpoint').value;
          const fileInput = document.getElementById('file');
          const output = document.getElementById('output');
          const preview = document.getElementById('preview');
          const gallery = document.getElementById('gallery');
          output.textContent = "Enviando...";
          preview.removeAttribute('src');
          preview.alt = "Sem imagem ainda";
          gallery.innerHTML = '<p class="muted">Carregando...</p>';

          if (!fileInput.files.length) {
            output.textContent = "Selecione uma imagem primeiro.";
            gallery.innerHTML = '<p class="muted">Sem imagens.</p>';
            return;
          }

          const formData = new FormData();
          formData.append('file', fileInput.files[0]);

          try {
            const resp = await fetch(endpoint, { method: 'POST', body: formData });
            const json = await resp.json();
            output.textContent = JSON.stringify(json, null, 2);

            // preview padrão (se existir image_base64)
            if (json.image_base64) {
              preview.src = json.image_base64;
              preview.alt = "Imagem anotada";
            } else {
              preview.removeAttribute('src');
              preview.alt = "Sem image_base64.";
            }

            // coleta TODAS as imagens no JSON
            let images = collectImages(json);
            images = naturalSortImages(images);
            renderGallery(images);
          } catch (e) {
            output.textContent = "Erro: " + e;
            preview.removeAttribute('src');
            preview.alt = "Erro ao carregar imagem.";
            gallery.innerHTML = '<p class="muted">Erro ao carregar imagens.</p>';
          }
        }
      </javascript>
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

# ---------------- Detecção de Soja (VGG/Custom) ----------------
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
        resultado = detectar_soja_multibox(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-multibox-V2/",
    tags=["Detecção de Soja V2"],
    summary="Detecta múltiplas caixas (multi model) e retorna imagem anotada V2",
    responses={400: {"model": ErrorResponse}}
)
async def detect_multibox_route_v2(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_soja_multibox_V2(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ---------------- YOLO (CPU) ----------------
@app.post(
    "/detect-yolo/",
    tags=["YOLO (CPU)"],
    summary="Detecta múltiplas caixas com YOLOv8 (CPU) e retorna imagem anotada",
    response_model=YOLOResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_yolo_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        out = predict_yolo(image, conf_threshold=0.25, imgsz=640)
        # out possui: boxes, boxes_norm, raw, image_base64
        return JSONResponse(content=out)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ---------------- YOLO V2 (CPU) GREEN MASK ----------------
@app.post(
    "/detect-yolo-v2/",
    tags=["YOLO (CPU)"],
    summary="Detecta múltiplas caixas com YOLOv8 (CPU) e retorna imagem anotada filtrando verde",
    response_model=YOLOResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_yolo_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        out = predict_yolo_V2(image, conf_threshold=0.25, imgsz=640)
        # out possui: boxes, boxes_norm, raw, image_base64
        return JSONResponse(content=out)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
# ---------------- Contagem / Cor ----------------
@app.post(
    "/count-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos (pipeline clássico com steps)",
    responses={400: {"model": ErrorResponse}}
)
async def count_objects(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = contar_objetos_pil(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/count-green-objects/",
    tags=["Contagem / Cor"],
    summary="Conta objetos verdes (com steps)",
    responses={400: {"model": ErrorResponse}}
)
async def count_green(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_objetos_verdes(image)
        return JSONResponse(content=resultado)  # agora retorna todas as imagens + total + boxes
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post(
    "/analyze-all/",
    tags=["Análises Completas"],
    summary="Análise combinada (verde + Harris + Shi-Tomasi) com steps e imagem anotada",
    responses={400: {"model": ErrorResponse}}
)
async def analyze_all(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = analisar_todos(image)  # retorna data URLs: 1_..., 2_..., ..., image_base64 + métricas
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-shi-tomasi/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Shi-Tomasi)",
    responses={400: {"model": ErrorResponse}}
)
async def detect_shi_tomasi_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_shi_tomasi(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-harris/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris)",
    responses={400: {"model": ErrorResponse}}
)
async def detect_harris_route(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        resultado = detectar_harris(image)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post(
    "/detect-features/",
    tags=["Detecção de Features"],
    summary="Detecta pontos (Harris + Shi-Tomasi)",
    responses={400: {"model": ErrorResponse}}
)
async def detect_features(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)

        resultado_harris = detectar_harris(image)
        resultado_tomasi = detectar_shi_tomasi(image)

        return {
            "pontos_detectados_harris": resultado_harris["pontos_detectados"],
            "pontos_detectados_tomasi": resultado_tomasi["pontos_detectados"],
            "image_base64_harris": resultado_harris["image_base64"],
            "image_base64_tomasi": resultado_tomasi["image_base64"]
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ===================== Run =====================
if __name__ == "__main__":
    # Em dev, se a sua GPU estiver causando OOM por conta do reload,
    # rode com CPU forçada em outros processos pesados. Para YOLO já é CPU.
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
