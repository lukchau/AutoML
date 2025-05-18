"""Модуль содержит классы для таблиц"""
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Index
from .database import Base


class User(Base):
    """Класс для таблицы "users

    Attributes:
        id: id пользователя
        email: email
        password: пароль
    """
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]

    __table_args__ = (
        Index("auth_idx", "email", "password"),
        Index("email_idx", "email")
    )
