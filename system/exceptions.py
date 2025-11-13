# system/exceptions.py

from fastapi import status

class AppException(Exception):
    """Classe base para exceções da aplicação."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

class ValidationError(AppException):
    """Erro de validação de dados de entrada."""
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

class AuthenticationError(AppException):
    """Erro de autenticação (senha incorreta, token inválido)."""
    def __init__(self, detail: str = "Credenciais inválidas."):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

class NotFoundError(AppException):
    """Recurso não encontrado (usuário, análise, etc.)."""
    def __init__(self, detail: str = "Recurso não encontrado."):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class FileProcessingError(AppException):
    """Erro durante o processamento de um arquivo (vídeo/imagem)."""
    def __init__(self, detail: str = "Falha ao processar o arquivo."):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

# Adicione outras exceções conforme necessário...