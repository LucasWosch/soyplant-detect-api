from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
#import tensorflow as tf
from object_counter import contar_objetos_pil
from green_detector import detectar_objetos_verdes
from feature_detector import detectar_harris, detectar_shi_tomasi
from full_analysis import analisar_todos

from kerasTrain.predict import predict_image  # Certifique-se de que esse arquivo existe

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        prediction = predict_image(image)
        confidence = round(prediction * 100, 2)
        label = "Pé de soja detectado" if prediction > 0.5 else "Nenhum pé de soja detectado"
        return JSONResponse(content={
            "label": label,
            "confidence_percent": confidence,
            "raw_prediction": round(prediction, 4)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/count-objects/")
async def count_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        total_objetos = contar_objetos_pil(image)

        return JSONResponse(content={
            "total_objetos_detectados": total_objetos
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return len(contornos)

@app.post("/count-green-objects/")
async def count_green(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        total = detectar_objetos_verdes(image)
        return JSONResponse(content={
            "total_verde_detectado": total
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/detect-shi-tomasi/")
async def detect_shi(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    total = detectar_shi_tomasi(image)
    return {"pontos_detectados": total}

@app.post("/detect-harris/")
async def detect_harris(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        total = detectar_harris(image)

        return JSONResponse(content={
            "pontos_detectados": total
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/detect-features/")
async def detect_features(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        total_harris = detectar_harris(image)
        total_tomasi = detectar_shi_tomasi(image)

        return JSONResponse(content={
            "pontos_detectados_harris": total_harris,
            "pontos_detectados_tomasi": total_tomasi
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/analyze-all/")
async def analyze_all(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        resultado = analisar_todos(image)

        return JSONResponse(content=resultado)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
