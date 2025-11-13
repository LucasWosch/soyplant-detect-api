from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict
from datetime import datetime

# ==============================
# Modelos de Resposta (Response)
# ==============================

class Token(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr  # ✅ Campo retornado na resposta da API

    class Config:
        from_attributes = True


class AnaliseResponse(BaseModel):
    id: int
    message: str
    contagem_total_unicos: int
    nome_arquivo_original: Optional[str] = None  # Adicionado para consistência

    class Config:
        from_attributes = True

class HistoricoAnaliseResponse(BaseModel):
    id: int
    nome_arquivo_original: str
    data_analise: datetime
    contagem_total_unicos: int
    latitude: Optional[float]
    longitude: Optional[float]
    local_texto: Optional[str]
    video_salvo_em: str

    class Config:
        from_attributes = True


# ==========================
# Modelo de Erro Padronizado
# ==========================

class ErrorResponse(BaseModel):
    """Modelo padrão para respostas de erro."""
    error: str = Field(..., description="Mensagem de erro principal.")
    details: Optional[Dict[str, str]] = Field(None, description="Detalhes adicionais sobre o erro (ex: campos inválidos).")


# ===============================
# Modelos de Requisição (Request)
# ===============================

class UserCreate(BaseModel):
    """Modelo para a criação de um novo utilizador com validações robustas."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=20,
        pattern="^[a-zA-Z0-9_]+$",
        description="Nome de usuário deve ser alfanumérico e ter entre 3 e 20 caracteres."
    )
    email: EmailStr
    password: str = Field(
        ...,
        min_length=8,
        max_length=500,
        description="A senha deve ter pelo menos 8 caracteres."
    )

    @validator('password')
    def validate_password_strength(cls, v):
        """Validador customizado para garantir a força da senha."""
        if not any(char.isdigit() for char in v):
            raise ValueError('A senha deve conter pelo menos um número.')
        if not any(char.isupper() for char in v):
            raise ValueError('A senha deve conter pelo menos uma letra maiúscula.')
        return v


class VideoAnalysisRequest(BaseModel):
    """Modelo para validar os dados do formulário de análise de vídeo."""
    latitude: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description="Latitude deve estar entre -90 e 90."
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude deve estar entre -180 e 180."
    )
    local_texto: Optional[str] = Field(
        None,
        max_length=255,
        description="Nome do local deve ter no máximo 255 caracteres."
    )


# Este modelo é mais um guia para as validações no controller,
# pois a validação de arquivo (tamanho, tipo) é feita programaticamente.
class VideoUploadConstraints(BaseModel):
    """Define as restrições para o upload de um arquivo de vídeo."""
    max_size_mb: int = Field(500, description="Tamanho máximo do arquivo em MB.")
    allowed_mime_types: List[str] = Field(
        default=['video/mp4', 'video/avi', 'video/mov', 'video/mkv'],
        description="Lista de tipos MIME permitidos para o vídeo."
    )