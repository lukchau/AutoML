from io import StringIO

from fastapi import (
    APIRouter,
    Cookie,
    HTTPException,
    Request,
    UploadFile,
    File,
)
from fastapi.responses import FileResponse
import logging
import os
import pandas as pd
import sys
from schemas.user import UserJWT
from api.helpers import decode_access_token
from AutoML.testScript import (
    load_data,
    preprocess_data,
    infer_target_column,
    infer_task_type,
    train_and_evaluate,
    validate_data,
    save_model
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Глобальные переменные для хранения состояния модели
trained_model = None
metrics = None
model_format = "joblib"


@router.post("/train")
async def train_model(
        file: UploadFile = File(...),
        token: str = Cookie(None, alias="users_access_token"),
):
    """Обучает модель на основе загруженного датасета"""
    # Проверка авторизации
    if not token:
        raise HTTPException(status_code=401, detail="Пользователь не авторизован")

    try:
        user_data = decode_access_token(token)
        user = UserJWT(**user_data)
        logger.info(f"Пользователь {user.id} запрашивает обучение модели.")
    except Exception as e:
        logger.error(f"Ошибка декодирования токена: {e}")
        raise HTTPException(status_code=401, detail="Неверный токен")

    # Проверка типа файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV-файлы поддерживаются")

    try:
        # Чтение файла с обработкой ошибок
        contents = await file.read()

        try:
            # Декодируем содержимое как UTF-8 (можно добавить другие кодировки)
            decoded_content = contents.decode('utf-8')
            df = pd.read_csv(StringIO(decoded_content))
        except UnicodeDecodeError:
            # Попробуем другие кодировки при необходимости
            raise HTTPException(status_code=400, detail="Неподдерживаемая кодировка файла")

    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        raise HTTPException(status_code=400, detail="Ошибка обработки файла")

    # Анализ данных
    try:
        target_column = infer_target_column(df)
        if not target_column:
            raise HTTPException(status_code=400, detail="Не удалось определить целевой столбец")

        task_type = infer_task_type(df, target_column)
        logger.info(f"Определён тип задачи: {task_type}")

        # Валидация данных
        validate_data(df, target_column)
        df_processed = preprocess_data(df, target_column)

        # Обучение модели
        global trained_model, metrics, model_format
        trained_model, metrics = train_and_evaluate(df_processed, target_column, task_type)
        save_model(trained_model, format=model_format)
        logger.info("Модель успешно обучена и сохранена.")

        return {
            "success": True,
            "message": "Модель обучена",
            "metrics": metrics,
            "target": target_column,
            "task_type": task_type
        }

    except ValueError as ve:
        logger.warning(f"Ошибка валидации данных: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обучения модели")


@router.get("/download")
async def download_model():
    """
    Скачивание обученной модели.
    """
    global trained_model
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Модель ещё не обучена")

    model_file = f"model.{model_format}"
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Файл модели не найден")

    return FileResponse(model_file, filename=model_file, media_type='application/octet-stream')
