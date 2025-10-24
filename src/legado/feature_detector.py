import cv2
import numpy as np
import base64
from PIL import Image

def _encode_bgr_to_data_url(bgr_img, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem anotada.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def detectar_harris(pil_image: Image.Image, salvar_path: str = None) -> dict:
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)

    # Harris Corner Detection
    dst = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Marcar os cantos com vermelho
    img_result = img_bgr.copy()
    img_result[dst > 0.01 * dst.max()] = [0, 0, 255]

    num_pontos = int(np.sum(dst > 0.01 * dst.max()))

    if salvar_path:
        cv2.imwrite(salvar_path, img_result)

    return {
        "pontos_detectados": num_pontos,
        "image_base64": _encode_bgr_to_data_url(img_result)
    }

def detectar_shi_tomasi(pil_image: Image.Image, salvar_path: str = None) -> dict:
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    corners = corners.astype(np.intp) if corners is not None else []

    img_result = img_bgr.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_result, (x, y), 4, (0, 255, 0), -1)

    if salvar_path:
        cv2.imwrite(salvar_path, img_result)

    return {
        "pontos_detectados": len(corners),
        "image_base64": _encode_bgr_to_data_url(img_result)
    }
