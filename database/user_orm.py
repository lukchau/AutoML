"""Модуль содержит класс с методами для работы с сущностью "Пользователь" в БД"""
from .helpers import get_password_hash, verify_password
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .database import session
from sqlalchemy import select
from .model import User
from schemas.user import User as UserModel
from fastapi import HTTPException


class Auth_obj:
    @staticmethod
    async def add_new_user(user: UserModel) -> int:
        """Добавляет пользователя и возвращает его ID"""
        async with session() as conn:
            new_user = User(
                email=user.email,
                password=get_password_hash(user.password)
            )

            try:
                conn.add(new_user)
                await conn.commit()
                await conn.refresh(new_user)
                return new_user.id

            except IntegrityError:
                await conn.rollback()
                raise HTTPException(409, "User already exists")

            except Exception as e:
                await conn.rollback()
                raise HTTPException(500, detail=e)

    @staticmethod
    async def authenticate(email: str, password: str) -> int | HTTPException:
        async with session() as conn:
            """Авторизировать пользователя
            
            Args:
                email: почта
                password: пароль
            """
            try:
                query = select(User).where(User.email == email)
                result = await conn.execute(query)
                user = result.scalar_one_or_none()

                if not user:
                    raise HTTPException(
                        status_code=404,
                        detail={"success": False, "message": "Пользователь не найден"}
                    )

                if not verify_password(password, user.password):

                    raise HTTPException(
                        status_code=401,
                        detail={"success": False, "message": "Неверный пароль"}
                    )

                return user.id

            except SQLAlchemyError as e:

                raise HTTPException(
                    status_code=500,
                    detail={"success": False, "message": f"Ошибка базы данных: {str(e)}"}
                )

    @staticmethod
    async def update_user_data(user_id, **data) -> None:
        """Обновить данные пользователя

        Args:
            user_id: идентификатор пользователя
            data: новые данные о пользователе
        """
        async with session() as conn:
            user = await conn.get(User, user_id)

            if user:

                for param, new_value in data.items():

                    if hasattr(user, param):
                        setattr(user, param, new_value)

            await conn.commit()

    @staticmethod
    async def delete_user(user_id) -> None:
        """Удалить пользователя

        Args:
            user_id: идентификатор пользователя
        """
        async with session() as conn:

            user_to_delete = await conn.get(User, user_id)

            if not user_to_delete:
                HTTPException(status_code=404, detail="user not found")

            await conn.delete(user_to_delete)
            await conn.commit()
