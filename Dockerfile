FROM python:3.9-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia arquivos para dentro do container
COPY . .

# Instala dependências do projeto
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando para iniciar a aplicação
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

