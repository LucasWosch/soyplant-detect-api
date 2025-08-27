from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # backbone leve
results = model.train(
    data='data.yaml',
    epochs=10,
    imgsz=640,
    batch=8,
    device='cpu',
    workers=0,
    optimizer='adamw',   # opcional
    cos_lr=True,         # opcional
    patience=20          # early stop opcional
)

# melhor checkpoint
model = YOLO('runs/detect/train/weights/best.pt')
