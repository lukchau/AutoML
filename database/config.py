"""Модуль содержит класс для подключения к БД"""
from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    """Класс с переменными окружения"""
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    HOST = os.getenv("HOST")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM")

    @property
    def DATA_BASE_AUTH_URL(self) -> str:
        """Метод для возвращения ссылки для подключения к БД"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@localhost:{self.POSTGRES_PORT}/db"
        # для запуска в докере хост ставить db для тестов localhost

settings = Settings()
