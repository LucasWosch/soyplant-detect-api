import os

# Caminho da pasta que deseja verificar
pasta = r"C:\Users\Gamer\PycharmProjects\soyplant-detect-api\data\v7\train\labels"

for arquivo in os.listdir(pasta):
    caminho_arquivo = os.path.join(pasta, arquivo)

    if os.path.isfile(caminho_arquivo):
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            conteudo = f.read()

        if not conteudo.startswith("0 "):
            # Ajusta o primeiro caractere para "1"
            novo_conteudo = "0" + conteudo[1:] if conteudo else "0"

            with open(caminho_arquivo, "w", encoding="utf-8") as f:
                f.write(novo_conteudo)

            print(f"O arquivo '{arquivo}' foi ajustado. Agora começa com: '{novo_conteudo[:20]}'")
