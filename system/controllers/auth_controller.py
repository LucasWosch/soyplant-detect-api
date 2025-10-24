from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from system.models.database_models import Utilizador
from system.models.pydantic_models import UserCreate
import system.auth as auth
from system.database import get_db_session


class AuthController:
    async def register(self, user_data: UserCreate, db: AsyncSession):
        # Verifica se o nome de utilizador ou email já existem
        query_user = select(Utilizador).where(Utilizador.username == user_data.username)
        query_email = select(Utilizador).where(Utilizador.email == user_data.email)

        result_user = await db.execute(query_user)
        if result_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nome de utilizador já registado."
            )

        result_email = await db.execute(query_email)
        if result_email.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Endereço de email já registado."
            )

        # Cria o novo utilizador com todos os campos
        hashed_password = auth.get_password_hash(user_data.password)
        novo_utilizador = Utilizador(
            username=user_data.username,
            email=user_data.email,  # ✅ Passa o email para o modelo do BD
            hashed_password=hashed_password
        )
        db.add(novo_utilizador)
        await db.commit()
        await db.refresh(novo_utilizador)

        return novo_utilizador

    async def login(self, form_data: OAuth2PasswordRequestForm, db: AsyncSession):
        user = await auth.authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Nome de utilizador ou password incorretos",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = auth.create_access_token(data={"sub": user.username})

        return {"access_token": access_token, "token_type": "bearer"}


auth_controller = AuthController()