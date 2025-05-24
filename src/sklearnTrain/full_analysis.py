import cv2
import numpy as np
from PIL import Image

def mostrar_reduzido(nome_janela, imagem, largura_fixa=500):
    altura, largura = imagem.shape[:2]
    proporcao = largura_fixa / float(largura)
    nova_altura = int(altura * proporcao)
    nova_img = cv2.resize(imagem, (largura_fixa, nova_altura))
    cv2.imshow(nome_janela, nova_img)

def analisar_todos(pil_image: Image.Image, salvar_path: str = None) -> dict:
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1. Blur e conversão para HSV
    blurred = cv2.GaussianBlur(img_bgr, (85, 85), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Máscara para regiões verdes
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # 3. Imagem isolada só com a parte verde
    img_verde = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    gray_verde = cv2.cvtColor(img_verde, cv2.COLOR_BGR2GRAY)

    # 4. Harris Corner
    harris = cv2.cornerHarris(np.float32(gray_verde), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    img_harris = img_verde.copy()
    img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    harris_pontos = int(np.sum(harris > 0.01 * harris.max()))

    # 5. Shi-Tomasi
    shi = cv2.goodFeaturesToTrack(gray_verde, maxCorners=500, qualityLevel=0.01, minDistance=10)
    shi_pontos = 0
    img_shi = img_verde.copy()
    if shi is not None:
        shi = shi.astype(np.intp)
        shi_pontos = len(shi)
        for corner in shi:
            x, y = corner.ravel()
            cv2.circle(img_shi, (x, y), 4, (0, 255, 0), -1)

    # 6. Contornos verdes
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]
    verde_pontos = len(contornos)

    # 7. Imagem final com tudo sobreposto
    resultado = img_bgr.copy()
    cv2.drawContours(resultado, contornos, -1, (255, 0, 0), 2)
    for corner in shi if shi is not None else []:
        x, y = corner.ravel()
        cv2.circle(resultado, (x, y), 3, (0, 255, 0), -1)
    resultado[harris > 0.01 * harris.max()] = [0, 0, 255]

    cv2.putText(resultado, f"Harris: {harris_pontos}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(resultado, f"Shi-Tomasi: {shi_pontos}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(resultado, f"Contornos verdes: {verde_pontos}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 8. Mostrar etapas
    mostrar_reduzido("1 - Original", img_bgr)
    mostrar_reduzido("2 - Verde segmentado", img_verde)
    mostrar_reduzido("3 - Harris", img_harris)
    mostrar_reduzido("4 - Shi-Tomasi", img_shi)
    mostrar_reduzido("5 - Máscara verde", mask)
    mostrar_reduzido("6 - Resultado final", resultado)
    cv2.destroyAllWindows()

    # 9. Salvar resultado, se necessário
    if salvar_path:
        cv2.imwrite(salvar_path, resultado)

    return {
        "shi_tomasi": shi_pontos,
        "harris": harris_pontos,
        "contornos_verdes": verde_pontos
    }
