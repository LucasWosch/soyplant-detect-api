# sklearnTrain/full_analysis.py
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

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

def analisar_todos(pil_image: Image.Image) -> Dict[str, Any]:
    # 1) Original (PIL -> BGR)
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 2) Blur + HSV
    blurred = cv2.GaussianBlur(img_bgr, (85, 85), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3) Máscara verde (ajustável)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # 4) Isola partes verdes
    img_verde = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    gray_verde = cv2.cvtColor(img_verde, cv2.COLOR_BGR2GRAY)

    # 5) Harris
    harris = cv2.cornerHarris(np.float32(gray_verde), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    thr_harris = 0.01 * (harris.max() if harris.size else 1.0)
    harris_map = (harris > thr_harris)
    img_harris = img_verde.copy()
    img_harris[harris_map] = [0, 0, 255]
    harris_pontos = int(np.count_nonzero(harris_map))

    # 6) Shi-Tomasi
    shi = cv2.goodFeaturesToTrack(gray_verde, maxCorners=500, qualityLevel=0.01, minDistance=10)
    shi_pontos = 0
    img_shi = img_verde.copy()
    if shi is not None:
        shi = shi.astype(np.intp)
        shi_pontos = len(shi)
        for corner in shi:
            x, y = corner.ravel()
            cv2.circle(img_shi, (x, y), 4, (0, 255, 0), -1)

    # 7) Contornos verdes
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]
    verde_pontos = len(contornos)

    # 8) Resultado final com tudo sobreposto
    resultado = img_bgr.copy()
    cv2.drawContours(resultado, contornos, -1, (255, 0, 0), 2)
    if shi is not None:
        for corner in shi:
            x, y = corner.ravel()
            cv2.circle(resultado, (x, y), 3, (0, 255, 0), -1)
    resultado[harris_map] = [0, 0, 255]

    cv2.putText(resultado, f"Harris: {harris_pontos}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(resultado, f"Shi-Tomasi: {shi_pontos}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(resultado, f"Contornos verdes: {verde_pontos}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 9) Boxes dos contornos (xyxy), opcional mas útil
    boxes: List[Dict[str, Any]] = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append({
            "x1": int(x), "y1": int(y),
            "x2": int(x + w), "y2": int(y + h),
            "area_px": float(cv2.contourArea(cnt))
        })
    boxes.sort(key=lambda b: b["area_px"], reverse=True)

    # 10) Imagens em dataURL (para o Playground mostrar todas)
    resp = {
        "1_original": _encode_bgr_to_data_url(img_bgr),
        "2_blurred": _encode_bgr_to_data_url(blurred),
        "3_hsv_as_bgr": _encode_bgr_to_data_url(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)),
        "4_mask": _encode_bgr_to_data_url(_to_bgr(mask)),
        "5_verde_segmentado": _encode_bgr_to_data_url(img_verde),
        "6_harris": _encode_bgr_to_data_url(img_harris),
        "7_shi_tomasi": _encode_bgr_to_data_url(img_shi),
        "8_resultado_final": _encode_bgr_to_data_url(resultado),
        "shi_tomasi": shi_pontos,
        "harris": harris_pontos,
        "contornos_verdes": verde_pontos,
        "boxes": boxes,
        # alias para preview padrão do Playground
        "image_base64": _encode_bgr_to_data_url(resultado)
    }
    return resp
