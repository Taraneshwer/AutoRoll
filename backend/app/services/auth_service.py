"""
Authentication Service providing Password Hashing, Verification, and JWT Token Management.
"""

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from app.database.models import User
from app.database.repositories.user_repository import UserRepository

SECRET_KEY = "autoroll_jwt_secret_key_change_in_production"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            parts = hashed_password.split("$")
            if len(parts) != 2:
                return False
            salt = bytes.fromhex(parts[0])
            key = bytes.fromhex(parts[1])
            new_key = hashlib.pbkdf2_hmac(
                "sha256", plain_password.encode("utf-8"), salt, 100000
            )
            return hmac.compare_digest(key, new_key)
        except Exception:
            return False

    @staticmethod
    def get_password_hash(password: str) -> str:
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return f"{salt.hex()}${key.hex()}"

    @staticmethod
    def create_access_token(
        data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": int(expire.timestamp())})

        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")
        )

        signature_input = f"{header_b64}.{payload_b64}".encode()
        signature = hmac.new(
            SECRET_KEY.encode(), signature_input, hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def authenticate_user(self, username: str, password: str) -> User | None:
        user = self.user_repo.get_by_username(username)
        if not user or not user.is_active:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
