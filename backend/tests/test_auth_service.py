"""
Unit tests for AuthService.
"""

from app.services.auth_service import AuthService


def test_password_hashing():
    pwd = "secretpassword123"
    hashed = AuthService.get_password_hash(pwd)

    assert hashed != pwd
    assert AuthService.verify_password(pwd, hashed) is True
    assert AuthService.verify_password("wrongpassword", hashed) is False


def test_jwt_token_creation():
    data = {"sub": "admin_user", "role": "ADMIN"}
    token = AuthService.create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 20
