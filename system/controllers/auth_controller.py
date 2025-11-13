from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Importação do modelo de dados e das exceções customizadas
from system.models.database_models import Utilizador
from system.models.pydantic_models import UserCreate
from system.exceptions import AuthenticationError
import system.auth as auth
from system.database import get_db_session


class AuthController:
    async def register(self, user_data: UserCreate, db: AsyncSession):
        """
        Registra um novo utilizador.
        A validação básica (username, email, password) já é feita pelo Pydantic no modelo UserCreate.
        """
        # Verifica se o nome de utilizador ou email já existem
        query_user = select(Utilizador).where(Utilizador.username == user_data.username)
        query_email = select(Utilizador).where(Utilizador.email == user_data.email)

        result_user = await db.execute(query_user)
        if result_user.scalar_one_or_none():
            # Usando nossa exceção customizada para um erro mais específico
            raise AuthenticationError(detail="Nome de utilizador já registado.")

        result_email = await db.execute(query_email)
        if result_email.scalar_one_or_none():
            raise AuthenticationError(detail="Endereço de email já registado.")

        # Cria o novo utilizador com a senha hasheada
        hashed_password = auth.get_password_hash(user_data.password)
        novo_utilizador = Utilizador(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password
        )
        db.add(novo_utilizador)
        await db.commit()
        await db.refresh(novo_utilizador)

        return novo_utilizador

    async def login(self, form_data: OAuth2PasswordRequestForm, db: AsyncSession):
        """
        Autentica um utilizador e retorna um token JWT.
        """
        user = await auth.authenticate_user(db, form_data.username, form_data.password)
        if not user:
            # Usando nossa exceção customizada para falha de autenticação
            raise AuthenticationError(detail="Nome de utilizador ou password incorretos.")

        access_token = auth.create_access_token(data={"sub": user.username})

        return {"access_token": access_token, "token_type": "bearer"}


auth_controller = AuthController()