# Ficheiro: src/yoloDetectionV3/routes/auth_routes.py

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

# Este import agora vai funcionar corretamente
from system.models.pydantic_models import Token, UserCreate, UserResponse
from system.controllers.auth_controller import auth_controller
from system.database import get_db_session

router = APIRouter(
    prefix="/api/v1",
    tags=["Autenticação"]
)

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session)
):
    return await auth_controller.login(form_data, db)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def registar_utilizador(
    user: UserCreate,
    db: AsyncSession = Depends(get_db_session)
):
    return await auth_controller.register(user, db)