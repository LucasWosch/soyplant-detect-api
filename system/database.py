# src/yoloDetectionV3/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from pydantic_settings import BaseSettings
import os

# Esta classe inteligente vai ler as variáveis do seu arquivo .env
class Settings(BaseSettings):
    db_user: str = "postgres"  # valor padrão
    db_password: str = "password"
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "tcc_soja"

    class Config:
        # Encontra o arquivo .env na pasta raiz do projeto
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

settings = Settings()

# Monta a URL de conexão para o SQLAlchemy
DATABASE_URL = (
    f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}@"
    f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

# Configura o "motor" do SQLAlchemy que gerencia a conexão
engine = create_async_engine(DATABASE_URL)
# Cria uma fábrica de sessões para interagir com o banco
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
# Uma classe base para nossos modelos de tabela
Base = declarative_base()

# Função que a nossa API vai usar para obter uma sessão do banco
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session