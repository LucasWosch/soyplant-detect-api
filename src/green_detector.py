# green_detector.py
import base64
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List

def _encode_bgr_to_data_url(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _as_bgr(img: np.ndarray) -> np.ndarray:
    # Converte imagens 1 canal (GRAY) ou HSV para BGR só para visualização consistente
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        # Heurística: se a soma do canal H (0..179) domina e S/V parecem válidos, tratamos como HSV
        # Mas é mais seguro passar explicitamente quando souber. Aqui detectamos por dtype/intervalo.
        h, s, v = cv2.split(img)
        if h.dtype == np.uint8 and s.dtype == np.uint8 and v.dtype == np.uint8 and (
            (h.max() <= 180 and s.max() <= 255 and v.max() <= 255)
        ):
            # Só converte se for provavelmente HSV; se já for BGR, conversão errada deixaria a cor estranha.
            # Como não temos flag, vamos apenas retornar como está. O chamador decide quando converter.
            return img
    return img

def detectar_objetos_verdes(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Retorna:
      - total: quantidade de contornos verdes
      - boxes: lista de {x1,y1,x2,y2,area_px}
      - imagens em dataURL: 1_original, 2_bgr_blur, 3_hsv_as_bgr, 4_mask, 5_mask_clean, 6_edges, 7_resultado
      - image_base64: alias de 7_resultado (para o Playground)
    """
    # 1) PIL -> BGR
    imagem_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 2) Blur em BGR
    blurred_bgr = cv2.GaussianBlur(imagem_bgr, (85, 85), 0)

    # 3) HSV
    hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

    # 4) Máscara (verde)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 5) Morfologia (limpeza)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 6) Canny na máscara limpa
    edges = cv2.Canny(mask_clean, threshold1=50, threshold2=150)

    # 7) Contornos e resultado
    contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) > 50]

    resultado = imagem_bgr.copy()
    cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)
    cv2.putText(
        resultado, f"Total: {len(contornos)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    # Boxes (xyxy) ordenados por área
    boxes: List[Dict[str, Any]] = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "area_px": float(cv2.contourArea(cnt))
        })
    boxes.sort(key=lambda b: b["area_px"], reverse=True)

    # Imagens em dataURL
    # hsv para visualização: converta para BGR só para exibir (não altera o pipeline)
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    resp = {
        "1_original": _encode_bgr_to_data_url(imagem_bgr),
        "2_bgr_blur": _encode_bgr_to_data_url(blurred_bgr),
        "3_hsv_as_bgr": _encode_bgr_to_data_url(hsv_vis),
        "4_mask": _encode_bgr_to_data_url(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
        "5_mask_clean": _encode_bgr_to_data_url(cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)),
        "6_edges": _encode_bgr_to_data_url(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
        "7_resultado": _encode_bgr_to_data_url(resultado),
        "total": len(contornos),
        "boxes": boxes,
        "image_base64": _encode_bgr_to_data_url(resultado)  # para o preview do Playground
    }
    return resp
