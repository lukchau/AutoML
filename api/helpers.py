"""Модуль с вспомогательными функциями для api"""
from jose import jwt
from datetime import datetime, timedelta, timezone
from AutoML.database.config import settings


def create_access_token(data: dict) -> str:
    """Создать токен

    Args:
        data: данные для токена
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    to_encode.update({'exp': expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, settings.ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Расшифровать токен

    Args:
        token: токен
    """
    return jwt.decode(token=token, key=settings.SECRET_KEY, algorithms=settings.ALGORITHM)