import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

# -------- Máscara de VERDE (HSV) --------
def mask_green_hsv(
    bgr: np.ndarray,
    low: Tuple[int, int, int] = (35, 25, 25),
    high: Tuple[int, int, int] = (90, 255, 255),
) -> np.ndarray:
    """
    Verde em HSV (intervalo padrão; ajuste conforme seu dataset).
    H: 35-90 (verde/verde-amarelado), S e V > ~25 p/ evitar cinzas/pretos.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(low, dtype=np.uint8)
    upper = np.array(high, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# -------- Refinos úteis --------
def refine_mask(mask: np.ndarray, k: int = 3, iterations: int = 1) -> np.ndarray:
    """Limpa ruídos e fecha buracos pequenos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask

def remove_small_blobs(mask: np.ndarray, min_area: int = 60) -> np.ndarray:
    """Remove componentes muito pequenos (pontinhos) que atrapalham."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):  # 0 é background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

# -------- Aplicação da máscara --------
def keep_colors_and_black_rest(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mantém cores originais onde mask==255 e zera (preto) no resto."""
    result = np.zeros_like(bgr)
    result[mask > 0] = bgr[mask > 0]
    return result

# -------- Pipeline de uma imagem (APENAS VERDE) --------
def process_image(
    bgr: np.ndarray,
    green_hsv=(35, 25, 25, 90, 255, 255),
    min_area=60,
    morph_k=3,
    morph_iter=1,
) -> np.ndarray:
    g_low = green_hsv[:3]
    g_high = green_hsv[3:]

    # Apenas verde
    m_green = mask_green_hsv(bgr, g_low, g_high)
    mask = m_green

    # Refino
    mask = refine_mask(mask, k=morph_k, iterations=morph_iter)
    mask = remove_small_blobs(mask, min_area=min_area)

    # Aplica (verde mantido; resto preto)
    out = keep_colors_and_black_rest(bgr, mask)
    return out

# -------- Lote (mantém estrutura p/ YOLO) --------
def process_folder(
    images_dir: str,
    out_dir: str,
    exts=(".jpg", ".jpeg", ".png"),
    **kwargs,
):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in images_dir.rglob("*"):
        if img_path.is_dir():
            continue
        if img_path.suffix.lower() not in exts:
            continue

        rel = img_path.relative_to(images_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] não consegui ler {img_path}")
            continue

        out = process_image(bgr, **kwargs)
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            print(f"[WARN] falha ao escrever {out_path}")
            continue

        count += 1

    print(f"✅ {count} imagens processadas de {images_dir} -> {out_dir}")
    print("ℹ️  Labels NÃO são copiadas aqui; use copy_labels() abaixo.")

def copy_labels(src_base: str, dst_base: str):
    """
    Copia a árvore 'labels' inteira de src_base para dst_base:
      src_base/labels/train -> dst_base/labels/train
      src_base/labels/valid -> dst_base/labels/valid
    """
    src_labels = Path(src_base) / "labels"
    dst_labels = Path(dst_base) / "labels"
    if dst_labels.exists():
        shutil.rmtree(dst_labels)  # apaga se já existir
    shutil.copytree(src_labels, dst_labels)
    print(f"✅ Labels copiadas de {src_labels} -> {dst_labels}")

if __name__ == "__main__":
    # Ajuste o caminho base do seu dataset (formato YOLO: images/train, labels/train, etc.)
    base = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/v7"
    base_filtered = f"{base}_filtered"

    # Processa IMAGENS (train e valid) criando árvore paralela em base_filtered
    process_folder(
        images_dir=f"{base}/train/images",
        out_dir=f"{base_filtered}/train/images",
        green_hsv=(35, 25, 25, 90, 255, 255),  # somente verde
        min_area=80,
        morph_k=3,
        morph_iter=1,
    )
    process_folder(
        images_dir=f"{base}/valid/images",
        out_dir=f"{base_filtered}/valid/images",
        green_hsv=(35, 25, 25, 90, 255, 255),  # somente verde
        min_area=80,
        morph_k=3,
        morph_iter=1,
    )

    # Copia LABELS (inteiras) para a árvore filtrada
    copy_labels(f"{base}/train", f"{base_filtered}/train")
    copy_labels(f"{base}/valid", f"{base_filtered}/valid")

    print("\n✅ Pronto! Estrutura esperada:")
    print(f"{base_filtered}/")
    print(" ├── train/")
    print(" │    ├── images/  (imagens filtradas com apenas VERDE)")
    print(" │    └── labels/  (labels copiadas 1:1)")
    print(" └── valid/")
    print("      ├── images/")
    print("      └── labels/")
