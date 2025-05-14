from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf

from predict import predict_image  # Certifique-se que esse caminho está correto

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler imagem enviada
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))

        # Rodar a predição
        prediction = predict_image(image)

        # Calcular confiança como porcentagem
        confidence = round(prediction * 100, 2)

        # Gerar uma resposta descritiva
        label = "Pé de soja detectado" if prediction > 0.5 else "Nenhum pé de soja detectado"

        return JSONResponse(content={
            "label": label,
            "confidence_percent": confidence,
            "raw_prediction": round(prediction, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Rodar o servidor na porta 8000
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
