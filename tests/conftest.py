"""Модуль содержит фикстуры для тестирования api ручек сервиса"""
import asyncio

from _pytest.fixtures import SubRequest
from pytest_asyncio import fixture
from pytest import mark
from sqlalchemy import select
from database.model import User as UserDB
from database.database import session
from database.user_orm import Auth_obj
from schemas.user import User


@fixture(scope="session")
def event_loop():
    """Переопределяем event loop для стабильности"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@fixture(params=['test@mail.com'])
async def delete_user_if_exist(request: SubRequest) -> None:
    """Фикстура. Удалить пользователя если он существует

    Args:
        request: email пользователя
    """
    async with session() as conn:
        query = select(UserDB).where(UserDB.email == request.param)
        result = await conn.execute(query)
        new_user = result.scalar_one_or_none()

        if new_user:
            await Auth_obj.delete_user(user_id=new_user.id)


@fixture(params=[('reg_user@mail.com', 'password')])
@mark.parametrize('delete_user_if_exist', ['reg_user@mail.com'], indirect=True)
async def register_user(request: SubRequest, delete_user_if_exist: None) -> None:
    """Фикстура. Зарегистрировать пользователя

    Args:
        request: почта и пароль пользователя
        delete_user_if_exist: удалить пользователя если имеется
    """
    user = User(email=request.param[0], password=request.param[1])
    new_user_id = await Auth_obj.add_new_user(user=user)

    yield new_user_id

    await Auth_obj.delete_user(user_id=new_user_id)
