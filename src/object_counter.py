import cv2
import numpy as np
from PIL import Image

def contar_objetos_pil(pil_image: Image.Image, salvar_path: str = None) -> int:
    # Converter PIL para OpenCV (RGB → BGR)
    original = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Suavização e limiarização
    blur = cv2.GaussianBlur(gray, (45, 45), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operações morfológicas
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contornos
    contornos, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]

    # Máscaras
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contornos, -1, 255, -1)
    mask_inv = cv2.bitwise_not(mask)

    # Blur no fundo
    blurred_background = cv2.GaussianBlur(original, (15, 15), 0)
    background_only = cv2.bitwise_and(blurred_background, blurred_background, mask=mask_inv)
    foreground_only = cv2.bitwise_and(original, original, mask=mask)
    final_image = cv2.add(background_only, foreground_only)

    # Contornos e texto final
    cv2.drawContours(final_image, contornos, -1, (0, 255, 0), 2)
    cv2.putText(final_image, f"Total: {len(contornos)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibir todas as etapas
    mostrar_reduzido("1 - Original", original)
    mostrar_reduzido("2 - Cinza", gray)
    mostrar_reduzido("3 - Blur", blur)
    mostrar_reduzido("4 - Threshold", thresh)
    mostrar_reduzido("5 - Opening", opening)
    mostrar_reduzido("6 - Máscara", mask)
    mostrar_reduzido("7 - Máscara Invertida", mask_inv)
    mostrar_reduzido("8 - Fundo Borrado", background_only)
    mostrar_reduzido("9 - Objetos sem Blur", foreground_only)
    mostrar_reduzido("10 - Final com contornos", final_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salvar imagem final, se solicitado
    if salvar_path:
        cv2.imwrite(salvar_path, final_image)

    return len(contornos)

def mostrar_reduzido(nome_janela, imagem, largura_fixa=500):
    altura, largura = imagem.shape[:2]
    proporcao = largura_fixa / float(largura)
    nova_altura = int(altura * proporcao)
    nova_img = cv2.resize(imagem, (largura_fixa, nova_altura))
    cv2.imshow(nome_janela, nova_img)

