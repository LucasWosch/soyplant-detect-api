import os
from pathlib import Path

# Configurações de caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
MODEL_PATH = str(PROJECT_ROOT / 'runs' / 'detect' / 'train4' / 'weights' / 'best.pt')
OUTPUT_DIR = str(PROJECT_ROOT / "videos_processados")

# Configurações YOLO
CONF_THRES = 0.20
IOU_THRES = 0.45
JPEG_QUALITY = 80

# Configurações SORT
SORT_MAX_AGE = 30
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3