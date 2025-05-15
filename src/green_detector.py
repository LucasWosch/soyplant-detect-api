import cv2
import numpy as np
from PIL import Image

def mostrar_reduzido(nome_janela, imagem, largura_fixa=500):
    altura, largura = imagem.shape[:2]
    proporcao = largura_fixa / float(largura)
    nova_altura = int(altura * proporcao)
    nova_img = cv2.resize(imagem, (largura_fixa, nova_altura))
    cv2.imshow(nome_janela, nova_img)

def detectar_objetos_verdes(pil_image: Image.Image, salvar_path: str = None) -> int:
    # Converter PIL para OpenCV BGR
    imagem_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ✅ Aplicar GaussianBlur antes de tudo
    blurred_bgr = cv2.GaussianBlur(imagem_bgr, (85, 85), 0)

    # Converter para HSV
    hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

    # Intervalo de cor verde (ajustável conforme o tom da planta)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Criar máscara e limpar com morfologia
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 🔍 Aplicar Canny na máscara limpa
    edges = cv2.Canny(mask_clean, threshold1=50, threshold2=150)

    # Encontrar contornos
    contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]

    # Resultado final
    resultado = imagem_bgr.copy()
    cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)
    cv2.putText(resultado, f"Total: {len(contornos)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar etapas
    mostrar_reduzido("1 - Original", imagem_bgr)
    mostrar_reduzido("2 - BGR Blur", blurred_bgr)
    mostrar_reduzido("3 - HSV", hsv)
    mostrar_reduzido("4 - Máscara Verde", mask)
    mostrar_reduzido("5 - Máscara Limpa", mask_clean)
    mostrar_reduzido("6 - Canny Edges", edges)
    mostrar_reduzido("7 - Resultado Final", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salvar se necessário
    if salvar_path:
        cv2.imwrite(salvar_path, resultado)

    return len(contornos)
