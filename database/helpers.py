"""Модуль содержит вспомогательные функции для работы с БД"""
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Получить захэшированный пароль

    Args:
        password: незахэшированный пароль
    """
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password) -> bool:
    """Сравнить пароли

    Args:
        plain_password: незахэшированный пароль
        hashed_password: захэшированный пароль
    """
    return pwd_context.verify(plain_password, hashed_password)
