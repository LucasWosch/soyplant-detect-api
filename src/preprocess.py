import cv2
import os
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Caminhos a partir da .env
image_folder = os.getenv('RAW_IMAGES_PATH')
sobel_folder = os.getenv('SOBEL_IMAGES_PATH')

# Verificação se as variáveis estão definidas
if not image_folder or not sobel_folder:
    raise ValueError("As variáveis de ambiente RAW_IMAGES_PATH ou SOBEL_IMAGES_PATH não foram definidas.")

# Criar a pasta de saída se não existir
if not os.path.exists(sobel_folder):
    os.makedirs(sobel_folder)

# Função para aplicar o filtro Sobel
def apply_sobel(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)

    return sobel_edges

# Processar imagens
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        sobel_image = apply_sobel(image_path)

        sobel_image_path = os.path.join(sobel_folder, f'sobel_{image_name}')
        cv2.imwrite(sobel_image_path, sobel_image)

        print(f'Imagem processada e salva: {sobel_image_path}')
