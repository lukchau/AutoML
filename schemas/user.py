"""Модуль содержит схемы для сущности "Пользователь"""
from pydantic import BaseModel, EmailStr, StrictStr, StrictInt, Field


class User(BaseModel):
    """Класс с моделью пользователя"""
    email: EmailStr = Field(description='email пользователя')
    password: StrictStr = Field(description='пароль')


class UserJWT(BaseModel):
    """Класс с моделью информации из jwt users_access_token"""
    id: StrictInt = Field(description='id пользователя в БД')
