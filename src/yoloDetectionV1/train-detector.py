# train_yolo.py
from ultralytics import YOLO

def main():
    # Carrega o modelo pré-treinado (YOLOv8n)
    model = YOLO("yolov8n.pt")

    # Treino (equivalente ao comando CLI)
    model.train(
        data="data.yaml",   # caminho do seu data.yaml
        epochs=50,          # épocas
        imgsz=640,          # tamanho da imagem
        batch=16,           # batch size
        device=0,           # GPU 0; use "cpu" se não tiver GPU
        workers=0           # 0 no Windows; pode usar >0 no Linux
    )

    # (Opcional) Avaliar no conjunto de validação
    model.val(data="data.yaml", imgsz=640, batch=16, device=0, workers=0)

    # (Opcional) Fazer uma inferência rápida após o treino
    # Troque "caminho/para/imagem.jpg" por um arquivo da sua base
    # results = model.predict(source="caminho/para/imagem.jpg", imgsz=640, device=0)

if __name__ == "__main__":
    main()
