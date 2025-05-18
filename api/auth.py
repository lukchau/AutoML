from fastapi import APIRouter, Response, Cookie, HTTPException
from database.user_orm import Auth_obj
from schemas.user import User, UserJWT
from fastapi.responses import JSONResponse
from .helpers import create_access_token, decode_access_token

router = APIRouter(prefix='/users', tags=['Authentication'])


@router.post("/register")
async def add_new_user(response: Response, new_user: User):

    user_id = await Auth_obj.add_new_user(new_user)
    access_token = create_access_token({'id': user_id})

    response.set_cookie(
        key='users_access_token',
        value=access_token,
        httponly=True,
        secure=True,
        samesite='lax',
        max_age=1800  # 30 минут
    )


@router.post('/login')
async def authenticate(response: Response, user: User):
    user_id = await Auth_obj.authenticate(user.email, user.password)
    access_token = create_access_token({'id': user_id})

    response.set_cookie(
        key='users_access_token',
        value=access_token,
        httponly=True,
        secure=True,
        samesite='lax',
        max_age=1800  # 30 минут
    )


@router.delete("/")
async def delete_user(
        token: str = Cookie(None, alias='users_access_token'), response: Response = None):

    if token:

        user = UserJWT.model_validate(decode_access_token(token=token))

        await Auth_obj.delete_user(user.id)

        response = JSONResponse(
            content={'message': 'Пользователь успешно удален'},
            status_code=200
        )

        response.delete_cookie(key='users_access_token')

        return response

    raise HTTPException(status_code=401, detail={'Пользователь не авторизован'})
