
# 🌿 SoyPlant Detect API

Uma API completa desenvolvida em **FastAPI** para análise de imagens com foco em detecção de **pés de soja**, contagem de objetos, identificação de áreas verdes e pontos de interesse utilizando técnicas clássicas de visão computacional e aprendizado de máquina.

---

O repositório contém um arquivo `TCC.postman_collection.json` com exemplos de requisições prontas para importar no [Postman](https://www.postman.com/).

---

## 🚀 Funcionalidades

- 🎯 Predição de presença de pé de soja com modelo treinado
- 🔢 Contagem de objetos com filtro Sobel
- 🌱 Detecção de regiões verdes (como folhas/plantações)
- 🧠 Detecção de características com:
  - Harris Corner
  - Shi-Tomasi
- 📊 Análise combinada com retorno estruturado

---

## 🧩 Tecnologias Utilizadas

- **FastAPI** para criação da API
- **OpenCV** para visão computacional
- **Pillow** para manipulação de imagens
- **TensorFlow** para predição com modelo treinado
- **NumPy** para processamento numérico

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/soyplant-detect-api.git
cd soyplant-detect-api
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute a API:

```bash
uvicorn main:app --reload
```

---

## 📁 Estrutura dos Endpoints

### `POST /predict/`

Realiza a predição de presença de pé de soja usando um modelo treinado.

#### Payload:

- Arquivo de imagem (`.jpg`, `.png`)

#### Resposta:

```json
{
  "label": "Pé de soja detectado",
  "confidence_percent": 97.52,
  "raw_prediction": 0.9752
}
```

---

### `POST /count-objects/`

Aplica filtro Sobel para contagem de objetos em imagens.

```json
{
  "total_objetos_detectados": 14
}
```

---

### `POST /count-green-objects/`

Detecta áreas verdes (ex: folhas) usando o espaço de cor HSV.

```json
{
  "total_verde_detectado": 6
}
```

---

### `POST /detect-shi-tomasi/`

Detecta pontos de interesse com o algoritmo Shi-Tomasi.

```json
{
  "pontos_detectados": 127
}
```

---

### `POST /detect-harris/`

Detecta cantos com o algoritmo Harris Corner.

```json
{
  "pontos_detectados": 982
}
```

---

### `POST /detect-features/`

Combina **Harris** e **Shi-Tomasi** na mesma análise.

```json
{
  "pontos_detectados_harris": 982,
  "pontos_detectados_tomasi": 127
}
```

---

### `POST /analyze-all/`

Executa análise completa com:

- Regiões verdes
- Harris Corner
- Shi-Tomasi

```json
{
  "shi_tomasi": 124,
  "harris": 895,
  "contornos_verdes": 7
}
```

---


