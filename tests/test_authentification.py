from sqlalchemy import select
from pytest import mark, param
from database.user_orm import Auth_obj
from database.helpers import verify_password
from schemas.user import User, UserJWT
from database.model import User as UserDB
from database.database import session
import requests
from api.helpers import decode_access_token


class TestAuthentication:
    @mark.asyncio
    @mark.parametrize('expected_status_code', [200])
    async def test_register(self, expected_status_code: int, delete_user_if_exist: None):
        user = User(email='test@mail.com', password='password')
        response = requests.request(
            method='POST',
            url='http://127.0.0.1:8080/users/register',
            json=user.model_dump(),
        )

        assert expected_status_code == response.status_code

        if response.status_code == 200:

            access_token = response.cookies.get('users_access_token')

            if not access_token:
                raise Exception('Cookie "users_access_token" не найден')

            async with session() as conn:
                query = select(UserDB).where(
                    UserDB.id ==
                    UserJWT.model_validate(decode_access_token(token=access_token)).id
                )
                result = await conn.execute(query)
                new_user = result.scalar_one_or_none()

                assert user.email == new_user.email
                assert verify_password(plain_password=user.password, hashed_password=new_user.password)

                await Auth_obj.delete_user(user_id=new_user.id)

    @mark.asyncio
    @mark.parametrize(
        'user_data, expected_status_code, register_user',
        [
            param(
                ('reg_user@mail.com', 'password'), 409, ('reg_user@mail.com', 'password'), marks=mark.critical
            ),
        ],
        indirect=['register_user'],
    )
    def test_negative_register(self, user_data: tuple[str, str], expected_status_code: int, register_user: None):
        response = requests.request(
            method='POST',
            url='http://127.0.0.1:8080/users/register',
            json=User(email=user_data[0], password=user_data[1]).model_dump(),
        )
        assert expected_status_code == response.status_code

    @mark.asyncio
    @mark.parametrize(
        'user_data, expected_status_code, register_user',
        [
            param(
                ('reg_user@mail.com', 'password'), 200, ('reg_user@mail.com', 'password'), marks=mark.critical
            ),
            (('reg_user@mail.com', 'pass'), 401, ('reg_user@mail.com', 'password')),
            (('None@mail.com', 'pass'), 404, ('reg_user@mail.com', 'password')),
        ],
        indirect=['register_user'],
    )
    async def test_login(self, user_data: tuple[str, str], expected_status_code: int, register_user: None):
        response = requests.request(
            method='POST',
            url='http://127.0.0.1:8080/users/login',
            json=User(email=user_data[0], password=user_data[1]).model_dump()
        )

        assert expected_status_code == response.status_code

        if expected_status_code == 200:

            async with session() as conn:

                new_user = await conn.get(UserDB, register_user)

                if new_user:
                    assert user_data[0] == new_user.email
                    assert verify_password(plain_password=user_data[1], hashed_password=new_user.password)

                else:
                    raise Exception('не удалось создать пользователя')
