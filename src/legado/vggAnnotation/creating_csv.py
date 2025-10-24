import json
import csv
import os
from PIL import Image

# Caminhos
json_path = "vgg_annotation.json"
images_folder = "../../data/v2/"  # Pasta onde estão suas imagens
output_csv = "annotations.csv"

# Carrega JSON do VIA
with open(json_path, 'r') as f:
    data = json.load(f)

# Abrir arquivo CSV de saída
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'class_id', 'x_center', 'y_center', 'width', 'height'])

    for key, value in data['_via_img_metadata'].items():
        filename = value['filename']
        regions = value['regions']

        # Tenta abrir a imagem para obter dimensões
        try:
            img_path = os.path.join(images_folder, filename)
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except:
            print(f"[AVISO] Não foi possível abrir {filename}. Pulando.")
            continue

        if len(regions) == 0:
            # Imagem negativa
            writer.writerow([filename, 0, 0, 0, 0, 0])
        else:
            # Imagens com box
            for region in regions:
                shape = region['shape_attributes']
                x = shape['x']
                y = shape['y']
                w = shape['width']
                h = shape['height']

                # Converte para formato YOLO-like (centro e tamanho normalizados)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                writer.writerow([filename, 0, round(x_center, 5), round(y_center, 5), round(width, 5), round(height, 5)])
