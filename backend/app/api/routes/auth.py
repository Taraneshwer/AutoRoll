"""
Authentication API Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.repositories.user_repository import UserRepository
from app.schemas.auth import LoginPayload, TokenResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=TokenResponse)
def login_for_access_token(payload: LoginPayload, db: Session = Depends(get_db)):
    user_repo = UserRepository(db)
    auth_service = AuthService(user_repo)

    user = auth_service.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.create_access_token(
        data={"sub": user.username, "role": user.role, "user_id": user.id}
    )

    return TokenResponse(
        access_token=token, token_type="bearer", username=user.username, role=user.role
    )
