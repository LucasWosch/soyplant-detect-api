import cv2
import numpy as np
import base64
from PIL import Image
from typing import Dict


def _encode_bgr_to_data_url(bgr_img, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def contar_objetos_pil(pil_image: Image.Image) -> Dict[str, str]:
    """
    Retorna todas as etapas (1 a 10) como imagens base64.
    """
    # Converter PIL para OpenCV (RGB → BGR)
    original = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Suavização e limiarização
    blur = cv2.GaussianBlur(gray, (45, 45), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Operações morfológicas
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contornos
    contornos, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]

    # Máscaras
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contornos, -1, 255, -1)
    mask_inv = cv2.bitwise_not(mask)

    # Blur no fundo
    blurred_background = cv2.GaussianBlur(original, (15, 15), 0)
    background_only = cv2.bitwise_and(
        blurred_background, blurred_background, mask=mask_inv
    )
    foreground_only = cv2.bitwise_and(original, original, mask=mask)
    final_image = cv2.add(background_only, foreground_only)

    # Contornos e texto final
    cv2.drawContours(final_image, contornos, -1, (0, 255, 0), 2)
    cv2.putText(
        final_image,
        f"Total: {len(contornos)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Retorno em formato YOLO-like
    return {
        "1_original": _encode_bgr_to_data_url(original),
        "2_gray": _encode_bgr_to_data_url(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        "3_blur": _encode_bgr_to_data_url(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)),
        "4_thresh": _encode_bgr_to_data_url(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)),
        "5_opening": _encode_bgr_to_data_url(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)),
        "6_mask": _encode_bgr_to_data_url(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
        "7_mask_inv": _encode_bgr_to_data_url(cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)),
        "8_background": _encode_bgr_to_data_url(background_only),
        "9_foreground": _encode_bgr_to_data_url(foreground_only),
        "10_final": _encode_bgr_to_data_url(final_image),
        "total": len(contornos),
    }
