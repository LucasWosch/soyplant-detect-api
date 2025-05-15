import cv2
import numpy as np
from PIL import Image

def mostrar_reduzido(nome_janela, imagem, largura_fixa=500):
    altura, largura = imagem.shape[:2]
    proporcao = largura_fixa / float(largura)
    nova_altura = int(altura * proporcao)
    nova_img = cv2.resize(imagem, (largura_fixa, nova_altura))
    cv2.imshow(nome_janela, nova_img)

def detectar_harris(pil_image: Image.Image, salvar_path: str = None) -> int:
    # Converter para BGR e float32
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)

    # Harris Corner Detection
    dst = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Marcar os cantos com vermelho
    img_result = img_bgr.copy()
    img_result[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Contar pontos (quantidade de pixels com resposta > threshold)
    num_pontos = np.sum(dst > 0.01 * dst.max())

    # Mostrar
    mostrar_reduzido("Harris Result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if salvar_path:
        cv2.imwrite(salvar_path, img_result)

    return int(num_pontos)

def detectar_shi_tomasi(pil_image: Image.Image, salvar_path: str = None) -> int:
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    corners = corners.astype(np.intp) if corners is not None else []

    img_result = img_bgr.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_result, (x, y), 4, (0, 255, 0), -1)

    mostrar_reduzido("Shi-Tomasi Result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if salvar_path:
        cv2.imwrite(salvar_path, img_result)

    return len(corners)

