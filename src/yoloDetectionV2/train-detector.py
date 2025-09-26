from ultralytics import YOLO
from datetime import datetime

def main(
    model_name: str = "yolov8n.pt",   # troque para "yolo12n.pt" se quiser YOLO12
    data: str = "data.yaml",
    device=0,                         # 0 = GPU; use "cpu" se quiser CPU
):
    start_time = datetime.now()
    print(f"🚀 Treinamento iniciado em: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model = YOLO(model_name)

    model.train(
        data=data,
        epochs=500,
        imgsz=640,
        batch=16,
        device=device,
        workers=0  # Windows: 0 é mais seguro
    )

    model.val(data=data, imgsz=640, batch=16, device=device, workers=0)

    end_time = datetime.now()
    print(f"🚀 Treinamento iniciado em: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Treinamento finalizado em: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Duração total: {end_time - start_time}")

if __name__ == "__main__":
    main()
1