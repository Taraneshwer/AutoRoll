"""
Authentication Pydantic Schemas.
"""

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class LoginPayload(BaseModel):
    username: str
    password: str
