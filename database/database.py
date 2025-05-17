"""Модуль содержит переменные и базовый класс для БД"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from .config import settings


engine = create_async_engine(url=settings.DATA_BASE_AUTH_URL)

session = async_sessionmaker(bind=engine, class_=AsyncSession)


class Base(DeclarativeBase):
    pass
