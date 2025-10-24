from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    email: EmailStr  # ✅ Campo obrigatório para criar um utilizador
    password: str = Field(..., min_length=8, max_length=500)


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